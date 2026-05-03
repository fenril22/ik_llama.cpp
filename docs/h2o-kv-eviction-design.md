# H2O KV Cache Eviction — 設計書

## 概要

H2O (Heavy Hitter Oracle) はattention weightに基づいてKV cacheのエントリを動的に
削除し、メモリ使用量を制限しつつ精度を維持するアルゴリズム。

本設計ではfattnカーネルを改造せず、定期的なスコアリング+evictionで実装する。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                    Decode Loop                            │
│                                                          │
│  for each token:                                         │
│    1. llama_decode(ctx, batch)  ← 通常のattention計算    │
│    2. if (step % EVICT_INTERVAL == 0 &&                  │
│           kv_used > kv_budget):                          │
│         a. h2o_compute_scores()  ← QK dot product       │
│         b. h2o_evict_lowest()    ← seq_rm + defrag      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 方式: 間引き実行スコアリング (方式D)

### なぜこの方式か

- fattn_kernel_t の型定義変更不要（7種のカーネルバリアント全てに影響回避）
- 通常decodeのパフォーマンスに影響なし（スコアリングは周期的にのみ実行）
- 既存インフラ (seq_rm, defrag) をそのまま活用
- attention maskが削除済みセルを自動的に -INF にするため、物理削除とロジカル削除が自然に分離

### 制約

- スコアは「最新Qとの関連度」の近似であり、過去のQ全てとの累積スコアではない
- 累積スコアが必要な場合はexponential moving average (EMA)で近似可能

## コンポーネント設計

### 1. CLI パラメータ

```
--kv-budget N        KVキャッシュの最大エントリ数 (default: -1 = 無制限)
--kv-sink N          先頭から無条件保持するエントリ数 (default: 2048)
--kv-evict-interval N  何decode step毎にevictionを実行 (default: 256)
```

### 2. スコアリングカーネル

**目的**: 最新のQuery (全head) と全KVエントリのdot productを計算し、
各KVエントリの「重要度スコア」を出力する。

**入力**:
- Q: 最新トークンのquery (全head) — shape [n_heads, head_dim]
- K: KV cacheのKey部分 — shape [n_kv, n_kv_heads, head_dim]（量子化型）

**出力**:
- scores: [n_kv] float — 各KVエントリのmax(over heads) QK dot product

**量子化KV対応の問題**:
KV cacheがq4_0やturbo3cの場合、Kは量子化されておりF16/F32として直接読めない。

**解決策**: `ggml_get_to_fp16_cuda(K->type)` を使ってF16に変換してからスコアリング。
これはMMAパスのdecodeと同じ変換関数。一時バッファが必要だが、eviction時のみ(256step毎)
なので許容可能。

```
スコアリング処理フロー:
1. K_f16 = alloc(n_kv * n_kv_heads * head_dim * sizeof(half))  // 一時バッファ
2. dequantize_row_xxx_cuda(K_data, K_f16, ...)                   // F16変換
3. k_h2o_compute_scores<<<...>>>(Q_f32, K_f16, scores, ...)      // スコア計算
4. free(K_f16)                                                    // 解放
```

**コスト見積もり** (128kコンテキスト, Qwen3.6-35B-A3B):
- F16変換: ~500MiB turbo3c → ~1280MiB F16一時バッファ → 変換 ~2ms
- スコアリング: 128k×8heads のQK dot → ~3ms
- 合計: ~5ms/256step = 0.02ms/step のamortizedオーバーヘッド

### 3. スコア蓄積 (EMA)

単一スナップショットではなく、過去のスコアを指数移動平均で蓄積:

```cpp
// eviction実行時:
for (int i = 0; i < n_kv; i++) {
    scores_ema[i] = alpha * scores_new[i] + (1 - alpha) * scores_ema[i];
}
// alpha = 0.3 (最新を重視するが過去も保持)
```

これにより「たまたま最新Qと関連が薄いが、過去には頻繁に参照されていた」KVが
即座に削除されることを防ぐ。

### 4. Eviction ロジック

```cpp
void h2o_evict(llama_context * ctx, int kv_budget, int kv_sink) {
    auto & cache = ctx->kv_self;
    int n_used = cache.used;
    
    if (n_used <= kv_budget) return;  // バジェット内なら何もしない
    
    int n_to_evict = n_used - kv_budget;
    
    // スコアでソートして最低スコアのKVを特定
    // ただしsink (先頭kv_sink個) は除外
    std::vector<std::pair<float, int>> scored_entries;
    for (int i = 0; i < cache.size; i++) {
        if (cache.cells[i].pos < 0) continue;  // 空セル
        if (cache.cells[i].pos < kv_sink) continue;  // sink保護
        scored_entries.push_back({scores_ema[i], i});
    }
    
    // スコア昇順ソート（低い=重要でない）
    std::sort(scored_entries.begin(), scored_entries.end());
    
    // 最低スコアからn_to_evict個を削除
    for (int i = 0; i < n_to_evict && i < scored_entries.size(); i++) {
        int cell_idx = scored_entries[i].second;
        llama_pos pos = cache.cells[cell_idx].pos;
        // 該当posのKVを全シーケンスから削除
        llama_kv_cache_seq_rm(ctx, -1, pos, pos + 1);
    }
    
    // メモリコンパクション
    llama_kv_cache_defrag(ctx);
}
```

### 5. Decode Loop統合

**examples/main/main.cpp への追加** (line 598付近の代替):

```cpp
// 既存のcontext-full処理を置き換え
if (params.kv_budget > 0 && n_past > 0 && n_past % params.kv_evict_interval == 0) {
    int kv_used = llama_get_kv_cache_used_cells(ctx);
    if (kv_used > params.kv_budget) {
        // 1. スコア計算
        h2o_compute_scores(ctx, scores_ema);
        // 2. Eviction
        h2o_evict(ctx, params.kv_budget, params.kv_sink);
        // 3. 位置調整は不要（maskが自動で-INFにする）
    }
}
```

**llama-server への追加** (examples/server/server-context.cpp):
同様のロジックをdecode後のフックに追加。

### 6. Q (最新Query) の取得方法

**問題**: decode後のQテンソルはGPU上の計算グラフの中間結果で、直接アクセスが困難。

**解決策**: llama_decode完了後に、最新トークンのembeddingを取得し、
Q projection (wq * x) を明示的に計算するか、
もしくは**最新トークンのKVエントリ自身のK値を使う**（K≈Qの近似）。

**最も簡単な近似**: K[latest_pos] を「最新Q」の代理として使い、
全KVエントリとのK-K dot product でスコアを計算する。
これはQとKが同じ空間にRoPE回転されているので合理的な近似。

```
score[i] = dot(K[latest_pos], K[i]) for all i
```

この方式なら**Qの取得問題が完全に解消**される。KV cacheのK部分だけ使えばよい。

### 7. K-K dot productスコアリング（推奨方式）

Q取得の複雑さを回避し、K同士のdot productでスコアリング:

```
入力: K cache (全KVエントリ) + 最新KVエントリの位置
処理:
  1. K[latest]をF16に変換 (or 既にF16なら直接使用)
  2. 各K[i]との dot product を計算
  3. scores[i] = dot(K[latest], K[i])

メリット:
  - Q projection不要
  - KV cacheから直接計算
  - 「最新のKeyに似たKey = 最近注目されやすいKey」の合理的な近似
```

## ファイル構成

```
ggml/src/ggml-cuda/h2o-score.cuh    — CUDAスコアリングカーネル宣言
ggml/src/ggml-cuda/h2o-score.cu     — CUDAスコアリングカーネル実装
src/llama-h2o.h                      — H2Oロジック (eviction, EMA, 統合)
src/llama-h2o.cpp                    — H2O実装
common/common.cpp                    — CLIパラメータ追加
examples/main/main.cpp               — decode loop統合
examples/server/server-context.cpp   — server統合
```

## 実装優先度

1. **Phase 1**: K-K dot productスコアリング + LRU eviction (fattn不要)
   - 量子化Kの1エントリのみF16変換 → 全Kと比較
   - 一時バッファ: head_dim * sizeof(half) = 512 bytes のみ
   - 最もシンプル、即座にテスト可能

2. **Phase 2**: EMAスコア蓄積 + CLIパラメータ
   - scores_ema配列の管理
   - --kv-budget / --kv-sink / --kv-evict-interval

3. **Phase 3**: サーバー統合 + 量子化K対応の最適化
   - server-context.cppへの組み込み
   - バッチdequant+dot productカーネル

## パフォーマンス見積もり

| 項目 | コスト | 頻度 | Amortized/step |
|------|--------|------|----------------|
| K[latest] dequant | ~0.01ms | 毎256step | ~0.00004ms |
| K-K dot product (128k entries) | ~3ms | 毎256step | ~0.012ms |
| ソート + eviction | ~0.5ms (CPU) | 毎256step | ~0.002ms |
| defrag | ~1ms | 毎256step | ~0.004ms |
| **合計** | **~4.5ms** | **毎256step** | **~0.018ms/step** |

通常decodeが ~36ms/step なので、**オーバーヘッド < 0.05%**。

## 期待効果

| 設定 | KVメモリ (200k) | 精度 | 速度 |
|------|:--:|:--:|:--:|
| q4_0 full (制限なし) | 1,125 MiB | +2.5% | 24 t/s |
| q4_0 + H2O budget=64k | ~360 MiB | +2.5〜3% | ~40 t/s |
| q4_0 + H2O budget=32k | ~180 MiB | +3〜5% | ~48 t/s |

## 実測ベンチマーク

**条件**: Qwen3.6-35B-A3B IQ3_S, RTX 3070 8GB, n-cpu-moe=30, flash-attn=1, c=2048, n=256
**H2O設定**: kv-budget=256, kv-sink=32, kv-evict-interval=32

| KV type | H2O | Decode (ms/tok) | Decode (t/s) | Prefill (t/s) |
|---------|-----|:--:|:--:|:--:|
| f16 | OFF | 14.92 | 67.04 | 129.5 |
| f16 | ON | 14.88 | 67.23 | 127.7 |
| q4_0 | OFF | 15.07 | 66.37 | 121.2 |
| q4_0 | ON | 15.07 | 66.37 | 123.4 |
| q4_1 | OFF | 15.04 | 66.51 | 130.5 |
| q4_1 | ON | 14.94 | 66.92 | 128.1 |
| turbo3c | OFF | 15.75 | 63.48 | 131.2 |
| turbo3c | ON | 15.63 | 63.98 | 133.7 |

**結論**: H2Oのスコアリング処理自体のオーバーヘッドは誤差範囲内（< 0.5%）。

### VRAM節約によるMoEオフロード改善（本来の効果）

**条件**: 42kトークンprefill + 256トークン生成, turbo3c KV, c=204800

| 設定 | KV (MiB) | n-cpu-moe | Prefill (t/s) | Decode (t/s) |
|------|:--:|:--:|:--:|:--:|
| H2O=OFF | 781 | 35 | 596 | 43.35 |
| H2O=ON budget=65k | 250 | 30 | 670 (+12%) | 44.67 (+3%) |
| H2O=ON budget=65k | 250 | 25 | **759 (+27%)** | **46.70 (+8%)** |

KV budget制限でVRAMが531MiB空き、MoE層をGPUに多く載せることで速度が向上。
`--n-cpu-moe`を35→25に減らせた（=10層分のMoE FFNがGPU実行に移行）。

**長文生成参考値** (turbo3c KV, 128kコンテキスト, H2O=OFF):
- Prefill: 546 t/s (1.83ms/tok), Decode: 28.1 t/s (35.5ms/tok)

## 注意事項

- **atomic float max**: CUDAのatomicMaxはint向け。float用には `__float_as_int` +
  `atomicMax` のtrick (正の値のみ正しく動作)。負のスコアには注意が必要。
- **マルチシーケンス**: ~~複数シーケンスがKVを共有する場合、evictionは全シーケンスに影響。
  server modeでは per-sequence evictionが必要。~~ **実装済み**: per-sequence eviction対応。
- **prefill中のeviction**: prefill中はevictionしない（全KVが必要）。decode開始後のみ。
- **KV cacheの穴**: seq_rm後にdefragしないとメモリが断片化。~~必ずセットで実行。~~
  **現状**: defrag+update がhybridモデル+量子化KVでクラッシュするためスキップ中。
  attention maskが穴を自動で-INFにするため動作に問題はないが、メモリ効率は低下。

## ハンドオフ情報

### 既存インフラの利用

| 機能 | 関数 | ファイル | 行 |
|------|------|----------|:--:|
| KVエントリ削除 | `llama_kv_cache_seq_rm()` | src/llama.cpp | 1602 |
| メモリコンパクション | `llama_kv_cache_defrag()` | src/llama.cpp | 1848 |
| セル使用数取得 | `cache.used` | src/llama-context.h | 48 |
| セル位置取得 | `cache.cells[i].pos` | src/llama-context.h | 16 |
| F16変換関数取得 | `ggml_get_to_fp16_cuda()` | ggml/src/ggml-cuda/convert.cu | 2020 |
| KVバッファポインタ | `cache.k_l[layer]->data` | src/llama.cpp | — |

### fattnカーネルの引数署名

```c
// fattn-vec-common.cuh line 20 (typedef fattn_kernel_t)
// 引数リスト: Q, K, V, mask, sinks, KV_min_max, dst, dst_meta,
//             scale, max_bias, m0, m1, n_head_log2, logit_softcap,
//             ne00-ne03, nb01-nb03, ne10-ne13, nb11-nb13, nb21-nb23,
//             ne31-ne33, nb31-nb33
// → この型定義を変更するとすべてのカーネルバリアントに影響するため触らない
```

### 量子化KVのストライド計算

```c
// fattn-common.cuh line 1023-1035
// quantized K → F16 変換時のストライド:
// const size_t bs = ggml_blck_size(K->type);  // e.g., 128 for turbo3c
// const size_t ts = ggml_type_size(K->type);  // e.g., 50 for turbo3c
// new_nb11 = old_nb11 * bs * sizeof(half) / ts;
```

### attention maskによる自動無効化

```c
// src/llama.cpp line 3596
// seq_rmで削除されたセル (pos == -1) はマスク生成時に自動的に -INF
// → fattnカーネルがsoftmaxで0weightにする → 追加の無効化処理不要
```

### KV_min_max の活用可能性

```c
// fattn-mma-f16.cuh line 1183
// flash_attn_mask_to_KV_min_max カーネルが既に存在
// attention maskから有効KV範囲を計算してカーネルに渡す
// → eviction後のdefragで連続化すれば、KV_min_maxは自然に正しくなる
```

### 最初に実装すべき最小構成

```
1. common/common.cpp: --kv-budget, --kv-sink パラメータ追加
2. src/llama-h2o.h/cpp: h2o_maybe_evict(ctx, step) 関数
   - KV使用量チェック
   - K[latest] vs K[all] のCPU dot product (GPU版は後で)
   - 最低スコアN個をseq_rm
   - defrag呼び出し
3. examples/main/main.cpp: decode loop内で h2o_maybe_evict() 呼び出し
```

**Phase 1はCPUのみ (GPUカーネルなし) で動作確認可能。**
K cacheをCPU側にコピーしてdot productを計算する。128kでも数msで完了する。
GPUカーネルは最適化フェーズで追加。
