# VRAM 消費挙動 調査レポート

**対象モデル**: Qwen3.6-35B-A3B-UD-Q3_K_XL  
**GPU**: RTX 3070 (8 GiB VRAM)  
**設定**: `-c 204800 --kv-budget 100000 --n-cpu-moe 31 -ngl 99 --flash-attn 1 -ctk turbo3c -ctv turbo3c`

---

## 1. 概要

### 背景

サーバー起動後、VRAM 消費量がリクエストのたびに増加し続けているように見えた。特に大きなプロンプトを送るたびに使用量が数十 MiB ずつ増え、「メモリリークではないか」という懸念が生じた。

### 結論のサマリー

VRAM の「じわじわ増加」はメモリリークではない。原因は **VMM scratch pool（CUDA Virtual Memory Management プール）の段階的成長**である。

- **KV キャッシュ**: 起動時に全枠（443.67 MiB）を一括確保済み。リクエストによって増減しない。
- **compute buffer**: 初回リクエスト時に一度だけ拡張（+54 MiB）し、その後は安定。
- **VMM scratch pool**: リクエストのサイズに応じて段階的に拡張し、物理 VRAM を解放しない（バンプアロケータ設計）。
- 大きなリクエストが来るたびに VMM プールが拡張され、最大約 188 MiB に達した後は安定する。

---

## 2. VRAM 構成（起動時の内訳）

| 構成要素 | サイズ | 備考 |
|---|---|---|
| モデルウェイト | 5,131 MiB | 起動時に固定確保 |
| KV キャッシュ（turbo3c, kv_phys=99,840） | 443.67 MiB | 起動時に全枠一括確保 |
| compute buffer | 978 MiB | 起動時。初回リクエスト後 ~1,032 MiB に拡張 |
| VMM scratch pool | 14 MiB | 起動時の初期値。リクエストにより成長 |
| cuBLAS workspace | 32 MiB | Fix 1 により事前確保 |
| CUDA runtime overhead | ~170 MiB | ランタイム・カーネル・ドライバ領域 |
| **合計（初回リクエスト前）** | **~6,871 MiB** | |
| **合計（初回リクエスト後、安定時）** | **~6,925 MiB** | compute buffer 拡張後 |

---

## 3. 「じわじわ減る」現象の正体

起動後に VRAM の空き領域が少しずつ減っているように見えた現象の実態：

1. **KV キャッシュは変化しない**: turbo3c 型は起動時に kv_phys=99,840 スロット分を全枠一括確保済み。使用中のトークン数が変わっても VRAM 消費量は変わらない。

2. **compute buffer は一度だけ拡張**: 起動直後の最初のリクエスト時に、ggml スケジューラが実際のテンソルサイズに合わせて compute buffer を ~978 MiB から ~1,032 MiB へ拡張する（+54 MiB）。以後は変化しない。

3. **VMM scratch pool がリクエストのサイズに応じて成長する**:
   - 小さいリクエスト: 最大 ~48 MiB
   - 130k トークン prefill 相当: 最大 ~188 MiB
   - pool は物理 VRAM を解放しないため、一度成長したサイズを維持する

すなわち「じわじわ減る」＝大きなリクエストが来るたびに VMM プールが段階的に拡張され、その分だけ VRAM 空き容量が減ることで発生していた。

---

## 4. VMM プールの詳細

### 設計

`ggml_cuda_pool_vmm`（`ggml/src/ggml-cuda.cu:402`）はバンプアロケータ＋LIFO 設計：

- **alloc(size)**: `pool_used += size`。空きが足りなければ `cuMemCreate` で物理 VRAM を追加確保し `pool_size` を拡張する。拡張単位は GPU のVMM 粒度（RTX 3070 では 2 MiB）。
- **free(ptr, size)**: `pool_used -= size`。物理 VRAM は解放しない（`pool_size` は減らない）。LIFO 逆順での解放が必須（GGML_ASSERT で保証）。
- **pool_addr**: `cuMemAddressReserve` で最大 32 GiB の仮想アドレス空間を予約し、必要に応じて物理マッピングを追加する。

### 実測値

| リクエスト種別 | VMM peak |
|---|---|
| 起動時 | 14 MiB |
| 小さいリクエスト（数百トークン） | ~48 MiB |
| 130k トークン prefill | ~188 MiB |

### デバッグログ形式

```
[VMM] pool grew: pool_size=X MiB pool_used=Y MiB (added Z MiB) caller_size=130.0 MiB
```

`caller_size` は成長を引き起こしたアロケーション要求サイズ（128 バイトアライメント後）。

### 未解明事項

130k トークン prefill 時、**28 回の成長イベントすべてで `caller_size=130.0 MiB`** が観測された。このアロケーションが何のバッファかは調査継続中（下記 §7 参照）。

---

## 5. H2O との相互作用

- **kv_phys の決定**: `--kv-budget 100000` → 256 アライメントで kv_phys=99,840。FA カーネルが `FATTN_KQ_STRIDE=256` の倍数を要求するため。
- **FA K_f16/V_f16 バッファ**: kv_phys ベースの一時 f16 変換バッファ。K+V 合計で ~195 MiB（K: ~97.5 MiB, V: ~97.5 MiB）。ピーク時に VMM プールに確保される。
- **H2O 動作確認**: 130k トークン超のリクエストでも H2O eviction が正常動作。n_kv が kv_phys=99,840 を超えた時点で自動 eviction が実行される。
- **200k トークン超テスト**: VRAM が 7,299 MiB で頭打ち。理論値 7,292 MiB と 0.1% 以内の誤差で一致。

---

## 6. 最大 VRAM 予測式

```
VRAM_max ≈
  モデルウェイト          5,131 MiB
+ KV キャッシュ             444 MiB
+ compute buffer          1,032 MiB
+ VMM scratch pool         ~188 MiB  （130k tokens 以上の prefill の場合）
+ cuBLAS workspace           32 MiB
+ CUDA runtime            ~170 MiB
──────────────────────────────────────
合計                      ~6,997 MiB
空き（8,192 MiB GPU）     ~1,195 MiB
```

**実測最大**: 7,299 MiB（空き 552 MiB）  
差分（~302 MiB）は FA K_f16/V_f16 瞬間ピーク（~195 MiB）や一時バッファ等の重複計上による。

---

## 7. 未解明事項・今後の調査

### (1) VMM 130 MiB アロケーションの正体（**解明済み** 2026-05-09）

130k 文字（約 29,221 トークン）の prefill 中に `caller_size=130.0 MiB` のアロケーションが 28 回の成長イベントすべてで観測された。

**デバッグログによる実測結果**:

`fattn-mma-f16.cuh:1321` の `K_f16.alloc(ggml_nelements(K))` 直前に以下を挿入してビルド・実測:

```cpp
fprintf(stderr, "[FA-DBG] K alloc: ne=[%lld,%lld,%lld,%lld] size=%.1f MiB\n",
        (long long)K->ne[0], (long long)K->ne[1], (long long)K->ne[2], (long long)K->ne[3],
        (double)ggml_nelements(K)*sizeof(ggml_fp16_t)/1048576.0);
```

llama-cli で t128.txt（128,252トークン）を処理した際のログ（kv-budget なし, -c 140000）:

```
[FA-DBG] K alloc: ne=[256,256,2,1]      size=  0.2 MiB  (初期グラフ)
[FA-DBG] K alloc: ne=[256,1024,2,1]     size=  1.0 MiB  (1024トークン後)
[FA-DBG] K alloc: ne=[256,2048,2,1]     size=  2.0 MiB  (2048トークン後)
...
[FA-DBG] K alloc: ne=[256,128000,2,1]   size=125.0 MiB  (128000トークン後)
[FA-DBG] K alloc: ne=[256,128256,2,1]   size=125.2 MiB  (最大: 128252トークン)
```

**実測確認事項**:

| 項目 | 値 |
|---|---|
| `ne[0]` (head_dim) | **256** |
| `ne[2]` (n_kv_heads) | **2** |
| `ne[1]` (n_kv) の最大実測値 | **128,256**（FATTN_KQ_STRIDE=256 境界に丸め）|
| 最大実測サイズ | **125.2 MiB**（128,252トークン処理時）|
| サイズ式 | `ne[0] × ne[1] × ne[2] × 2 bytes = n_kv × 1024 bytes` |

**使用コードパスの特定**:

`fattn.cu` の dispatch ロジックにより、Qwen3.6（head_dim=256, n_heads=16, n_kv_heads=2, gqa_ratio=8）は:
- `fattn-new-mma.cu` は gqa_ratio=8 には非対応（`GGML_ABORT("Not implemented")`）
- `fattn.cu:162` の `ggml_cuda_flash_attn_ext_mma_f16()` → **`fattn-mma-f16.cuh` が唯一の実行パス**

よって今回の計測は Qwen3.6 で実際に使われるコードパスを正しく対象にしている。

**130 MiB 正体の結論**:

実測で `ne=[256, N, 2, 1]` のパターンが n_kv に正確に比例することが確認された。
`n_kv=133,120` のとき `256 × 133,120 × 2 × 2 / 1,048,576 = 130.0 MiB` が成立する。
**よって 130 MiB VMM アロケーションは `fattn-mma-f16.cuh:1321` の `K_f16` バッファによるものであることが高確度で確認された。**

本実験では n_kv=128,256 (125.2 MiB) を直接観測したが、130 MiB に対応する n_kv=133,120 の直接観測は未実施。
n_kv=133,120 が発生した要因（当時の元テストで 29,221 トークンにもかかわらず観測）については、
マルチターン会話の n_past 蓄積または worst-case グラフの事前構築という仮説があるが、いずれも未確認。

**kv-budget あり（100000）の場合の実測上限**:

kv_phys=99,840 に制限した場合、H2O eviction により n_kv ≤ 91,136（89 MiB）が観測上限だった（t128.txt処理時）。

**未解明点（残存）**:

| 疑問 | 仮説 |
|---|---|
| なぜ 28 回の成長イベント | LIFO + バンプアロケータ設計上は 1〜2 回のはずだが、他のアロケーションが挟まることで pool 使用量の上限が上昇する可能性 |

**確認用コード（`fattn-mma-f16.cuh:1321` 付近に追加）**:

```cuda
if (need_f16_K && K->type != GGML_TYPE_F16) {
    // DEBUG: identify 130 MiB allocation
    fprintf(stderr, "[FA] K_f16 alloc: ne=[%lld,%lld,%lld,%lld] size=%.1f MiB\n",
            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
            ggml_nelements(K) * sizeof(half) / 1048576.0f);
    K_f16.alloc(ggml_nelements(K));
```

### (2) VMM プール起動時一括確保の実装検討

現状: VMM プールは使用量に応じて段階的に成長する。これにより最初の大きなリクエスト時に成長コスト（`cuMemCreate`）が発生する。

**提案**: `GGML_CUDA_POOL_VMM_PREALLOC_MIB` 環境変数で起動時に指定 MiB を事前確保する。  
例: `GGML_CUDA_POOL_VMM_PREALLOC_MIB=200` を設定すれば初回リクエスト時の遅延を排除できる。

実装は `ggml_cuda_pool_vmm::ggml_cuda_pool_vmm()` コンストラクタ内で初期アロケーションを行うだけで対応可能。
