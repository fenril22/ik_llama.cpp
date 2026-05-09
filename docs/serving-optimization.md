# サービング最適化ガイド（RTX 3070 / Qwen3.6-35B-A3B）

## 1. 問題の経緯

- 長時間稼働でVRAMが徐々に増加してOOMクラッシュ
- cublasSgemm CUBLAS_STATUS_INTERNAL_ERROR

## 2. VRAMが増加する根本原因（2つ）

### 原因1: Compute Buffer high-water mark

- ファイル: `ggml/src/ggml-alloc.c:228`
- `alloc->max_size = MAX(alloc->max_size, ...)` が単調増加のみ
- リクエストのバッチサイズが大きくなるたびに数十〜数百MiB単位で増加、縮小しない
- n_ubatch=1024時: 最大~2,580 MiB

### 原因2: VMM Pool high-water mark

- ファイル: `ggml/src/ggml-cuda.cu`
- `pool_size`は物理ページを解放しない（`free`は`pool_used`を減らすだけ）
- 大きなprefillが来るたびに拡張、以後永続

## 3. 実装した対策

### 対策1: 起動時ウォームアップ（Compute Buffer固定化）

- ファイル: `src/llama.cpp`（`llama_init_from_model`末尾）
- 起動時にn_ubatchサイズでダミーdecodeを1回実行
- Compute Bufferが起動時に最大値で確定、以後増えない
- ログ: `llama_init_from_model: warmup complete, compute buffer fixed at max size (n_ubatch=512)`

### 対策2: VMM Poolプリアロケーション

- ファイル: `ggml/src/ggml-cuda.cu`（`ggml_cuda_pool_vmm`コンストラクタ）
- 環境変数 `GGML_CUDA_POOL_VMM_PREALLOC_MiB`（デフォルト200 MiB）で事前確保
- ログ: `ggml_cuda_pool_vmm[0]: pre-allocated 200 MiB`

### 対策3: cuBLASワークスペース事前確保

- ファイル: `ggml/src/ggml-cuda/common.cuh`
- 32 MiBのワークスペースを起動時に確保（CUBLAS_STATUS_INTERNAL_ERROR防止）

## 4. パラメータ最適化の知見

### -b と -ub の役割

- `-b (n_batch)`: 論理バッチ上限。複数スロットをまとめてスケジューリングする単位
- `-ub (n_ubatch)`: 物理バッチ上限。Compute Bufferのサイズを決める（線形比例）
- VRAMを減らすには `-ub` を下げる

**実測値（Qwen3.6-35B-A3B, RTX 3070）**:

| n_ubatch | Compute Buffer | pp速度低下 | tg速度低下 |
|---|---|---|---|
| 1024 | ~2,580 MiB | 基準 | 基準 |
| 512 | ~732 MiB | -0.5% | -1.1% |
| 256 | ~490 MiB | -20.6% | -1.5% |

→ **n_ubatch=512が最適**（速度ほぼ変わらず、VRAM 1,848 MiB削減）

### -c (context size) とH2Oの関係

- H2O（`--kv-budget`）が有効な場合、物理KV = `min(n_ctx, kv_budget)`
- `-c` を増やしてもVRAMは増えない（kv_budget上限で物理確保が頭打ち）
- np=2の場合: `n_ctx_slot = n_ctx / np` → `-c`を増やすとスロットあたりのコンテキストが伸びる

### --n-cpu-moe の効果

- MoEレイヤーをCPUにオフロードする層数
- 1層あたりVRAM約24 MiB削減、tg速度約1 t/s低下
- GPU SM利用率がdecode中53〜55%止まりなのはCPUがボトルネックのため

| n-cpu-moe | tg (t/s) | VRAM(np=1) | 備考 |
|---|---|---|---|
| 31 | 58.6 | ~6,383 MiB | 旧設定 |
| 28 | 62.6 | ~7,377 MiB | 現在設定 |
| 26 | 64.0 | OOM（np=2不可） | |

## 5. スロット数（--parallel/-np）の知見

### KVキャッシュとスロットの関係

- Attention KV: スロット間で**共有**（スロット増やしてもAttention KV VRAMは変わらない）
- SSM状態（Qwen3固有）: スロット数に比例（+62.8 MiB/スロット）
- `n_ctx_slot = n_ctx / np`（スロットを増やすと1スロットあたりのコンテキストが短くなる）

### Agentワークロードでのスループット実測（マトリクス）

| パターン | np1/moe31 | np1/moe28 | np2/moe31 | np2/moe28 |
|---|---|---|---|---|
| 直列5連続 | 54.8 | 55.6 | 54.7 | 55.9 |
| 並列2同時 | 55.5 | **56.9** | 42.6 ★ | 41.7 ★ |
| 並列4同時 | 55.4 | **57.1** | 44.1 ★ | 45.0 ★ |
| ToolWait+subagent | 56.6 | **58.3** | 48.2 ★ | 48.1 ★ |
| ネスト4層 | 52.9 | **54.1** | 51.4 | 53.6 |

★ = np=1より遅い

**結論**: llama.cppのキュー処理がバッチングより効率的。subagent/tool call多用環境でも**np=1が最速**。

## 6. FA K_f16/V_f16バッファについて

- fattn-mma-f16.cuh でK/VをFP16に変換する際に確保
- サイズ: `ne=[256, n_kv, 2, 1]` → n_kv × 1024 bytes
- H2O有効時: n_kv ≤ kv_phys ≈ 99,840 → 最大~89 MiB
- H2O無効時: n_kv = 実際の累積トークン数 → 最大~130 MiB
- 推論ごとにVMM poolからalloc/freeされる（high-water markが残る）

## 7. 現在の推奨設定（2026-05-09時点）

```bash
# BASE_FLAGS
-b 2048 -ub 512
-ctk turbo3c -ctv turbo3c
-ngl 99 --flash-attn 1

# INF_FLAGS
--kv-budget 102400
-c 204800
--n-cpu-moe 28
-np 1

# 起動時VRAM: ~7,571 MiB（ウォームアップ込み）
# 空きVRAM: ~279 MiB
# tg速度: ~62.6 t/s
# pp速度: ~940 t/s
```

## 8. VRAMが増えないことの確認方法

```bash
# 起動後のVRAM
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# warmupログの確認
grep 'warmup\|prealloc' server.log

# リクエスト前後でVRAMが変化しないことを確認
# 期待値: 初回+数十MiB（KVキャッシュのみ）、2回目以降ゼロ増加
```
