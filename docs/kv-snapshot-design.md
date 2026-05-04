# KV Snapshot Store — 設計書

## 背景

### 問題

llama-serverの`cache_prompt`はスロット内のKV prefix再利用のみ。
マルチAgent環境では、スロットが別リクエストに奪われるとKVが消失し、
次のターンで全会話のフルprefill（数分）が必要になる。

```
AgentA: 128k会話 → スロット0にKV → prefill 3分
AgentB: スロット0を使用 → AgentAのKV消失
AgentA: 次のターン → フルprefill 3分やり直し  ← ここが問題
```

### 解決策

スロットの外にKV stateのスナップショットをRAM/ディスクに保持し、
スロット復帰時に即座にrestoreする。

```
AgentA: 完了 → snapshot save (19ms) → スロット解放
AgentB: スロット使用
AgentA: 再開 → snapshot restore (21ms) → 差分prefillのみ  ← 3分→21ms
```

## 実測データ

Qwen3.6-35B-A3B IQ3_S, RTX 3070, turbo3c KV, budget=65k

| 項目 | 値 |
|------|:--:|
| Snapshot size (65k cells) | 313 MB |
| Save速度 | 19 ms (17.3 GB/s) |
| Restore速度 | 21 ms (15.6 GB/s) |
| フルprefill (128k tokens) | 206 秒 |
| **Restore vs Prefill** | **10000倍速** |

### budgetとsnapshotサイズ

| budget | Snapshot | RAM 21GB空きで同時キャッシュ |
|:--:|:--:|:--:|
| 65k | ~313 MB | ~67会話 |
| 100k | ~478 MB | ~44会話 |

## アーキテクチャ

```
┌──────────────────────────────────────────────────────┐
│                   llama-server                         │
│                                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│  │ Slot 0  │  │ Slot 1  │  │ Slot N  │  (GPU KV)     │
│  └────┬────┘  └────┬────┘  └────┬────┘               │
│       │            │            │                      │
│  ┌────▼────────────▼────────────▼────┐                │
│  │         Snapshot Store             │  (CPU RAM)     │
│  │                                    │                │
│  │  conv_hash_A → snapshot_A (313MB)  │                │
│  │  conv_hash_B → snapshot_B (313MB)  │                │
│  │  conv_hash_C → snapshot_C (313MB)  │                │
│  │  ...                               │                │
│  │  LRU eviction when RAM pressure    │                │
│  └────────────────────────────────────┘                │
└──────────────────────────────────────────────────────┘
```

## フロー

### リクエスト到着時

```
1. messages のトークン列を計算
2. token prefix の hash を計算 → conv_hash
3. Snapshot Store で conv_hash を検索
   → ヒット:
     a. snapshot から prefix_len を取得
     b. スロットの KV を clear
     c. llama_state_seq_set_data() で restore (21ms)
     d. messages[prefix_len:] だけ差分 prefill
   → ミス:
     a. フル prefill
4. decode (生成)
```

### レスポンス完了時

```
1. 現在の token 列の hash → conv_hash
2. llama_state_seq_get_data() で KV を save (19ms)
3. Snapshot Store に conv_hash → snapshot を格納
4. 既存の同 conv_hash エントリは上書き
```

### メモリ管理

```
Snapshot Store の合計サイズが上限を超えた場合:
  → LRU (最終アクセスが最も古い) スナップショットを削除

上限の決定:
  max_snapshot_mem = (total_ram - model_size - pinned - os_reserve)
  例: 35GB - 11GB - 14GB - 3GB = 7GB → 最大22会話 (313MB each)

設定パラメータ:
  --kv-snapshot-max-mem N    最大RAM使用量 (MiB, default: 0 = 無効)
  --kv-snapshot-dir PATH     ディスクベースの場合のパス (tmpfs推奨)
```

## conv_hash の設計

### 要件

- 同じ会話の続きを正確に識別する
- prefix match: 「前回の会話 + 新メッセージ」のパターンを検出

### 方式

```
conv_hash = hash(tokens[0..n_tokens-1])

ただしprefix match が必要なので、単一hashでは不十分。
代わりに token prefix の段階的 hash を使用:

snapshot = {
    tokens: [t0, t1, ..., tn],    // 保存時のトークン列
    hash:   hash(tokens),          // 完全一致用
    data:   [KV state bytes],      // llama_state_seq_get_data() の出力
    last_access: timestamp,        // LRU用
}

検索時:
  1. 完全一致: hash(request_tokens) == snapshot.hash → restore全体
  2. Prefix match: request_tokens[0..m] == snapshot.tokens[0..m]
     → restore snapshot → request_tokens[m+1..] を差分prefill
```

### サーバーの既存 prefix match との統合

server-context.cpp の `get_common_prefix()` (line 3490) が既にトークン列の
prefix match を行っている。これを Snapshot Store の検索にも再利用:

```
1. リクエストのトークン列を計算
2. 全スナップショットに対して get_common_prefix() で最長一致を検索
3. 最長一致のスナップショットを restore
4. 差分を prefill
```

## 実装コンポーネント

### 1. SnapshotStore クラス (新規)

```cpp
// src/llama-snapshot-store.h

struct kv_snapshot {
    std::vector<llama_token> tokens;     // 保存時のトークン列
    std::vector<uint8_t>     kv_data;    // llama_state_seq_get_data() output
    int64_t                  last_access; // LRU timestamp
    llama_seq_id             seq_id;     // 保存元のseq_id
};

class kv_snapshot_store {
public:
    // 設定
    void set_max_memory(size_t max_bytes);

    // 保存: conversation の KV state を snapshot に保存
    void save(llama_context * ctx, llama_seq_id seq_id,
              const std::vector<llama_token> & tokens);

    // 検索+復元: tokens に最長一致する snapshot を restore
    // 戻り値: restore されたトークン数 (0=ミス)
    int restore(llama_context * ctx, llama_seq_id seq_id,
                const std::vector<llama_token> & tokens);

    // 統計
    size_t total_memory() const;
    int    count() const;

private:
    std::vector<kv_snapshot> snapshots_;
    size_t max_memory_ = 0;
    size_t used_memory_ = 0;

    void evict_lru();
    int  find_best_prefix(const std::vector<llama_token> & tokens);
};
```

### 2. サーバー統合

```
server-context.cpp:

  slot取得時 (launch_slot_with_task):
    snapshot_store.restore(ctx, slot.id, prompt_tokens)
    → 一致分は prefill skip、差分だけ処理

  slot解放時 (release_slot):
    snapshot_store.save(ctx, slot.id, slot.cache_tokens)

  初期化時:
    params_base.kv_snapshot_max_mem > 0 なら snapshot_store を初期化
```

### 3. CLI パラメータ

```
--kv-snapshot-max-mem N    スナップショットストアの最大RAM (MiB)
                           0 = 無効 (default)
                           例: 4096 = 4GB → ~13会話分
```

## 既存APIの利用

| 操作 | API | ファイル |
|------|-----|---------|
| KV save | `llama_state_seq_get_data()` | llama.h:972 |
| KV restore | `llama_state_seq_set_data()` | llama.h:979 |
| サイズ取得 | `llama_state_seq_get_size()` | llama.h:965 |
| ファイルI/O | `llama_state_seq_save_file()` | llama.h:987 |

## H2O との組み合わせ

```
H2O budget=65k:
  - KV cache は常に 65k cells 以下
  - Snapshot も 65k cells 分 (~313MB)
  - restore 後の KV は budget 内に収まっている
  - 差分 prefill 中も h2o_ensure_budget() が動作

H2O なし:
  - KV cache は n_ctx 分 (200k cells)
  - Snapshot が ~760MB に膨張
  - 同時キャッシュ数が減少
  → H2O 併用を推奨
```

## 段階的実装

### Phase 1: SnapshotStore + CLI テスト
- kv_snapshot_store クラス実装
- llama-cli に --kv-snapshot-max-mem を追加
- 手動 save/restore のテスト

### Phase 2: サーバー統合
- server-context.cpp にsave/restore を組み込み
- slot lifecycle への統合
- prefix match による部分 restore

### Phase 3: 最適化
- 差分 snapshot (前回との差分だけ保存)
- mmap ベースの大容量ストア
- 複数プロセス間での snapshot 共有

## リスクと制約

- **モデル/パラメータ不一致**: 異なるモデルや量子化設定のsnapshotは復元不可。
  snapshot にモデルhashを含めて検証する必要がある。
- **H2O eviction後の一貫性**: evictされたトークンのKVは消失しているので、
  snapshot復元後のKVはeviction済みの状態。精度には影響するが動作は正常。
- **hybrid (Mamba+Attention) モデル**: recurrent stateも保存対象。
  `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` でSSM stateのみ保存するオプションあり。
- **メモリ断片化**: 大量のsnapshotの確保/解放でRAMが断片化する可能性。
  プール allocator の検討が必要かも。
