# cublasSgemm CUBLAS_STATUS_INTERNAL_ERROR クラッシュ再現・検証テスト手順書

> 作成日: 2026-05-08  
> 対象ビルド: ik_llama.cpp build=4494 (commit bb8e378f)  
> 対象サーバー: vllm ホスト、ポート 8080

> **前提: curl コマンドはすべて vllm ホスト上で実行する**  
> ローカルマシンから `vllm` が DNS/hosts で解決できない場合は、各 `curl` コマンドを以下の形式で実行すること:  
> `ssh vllm "curl -s http://localhost:8080/..."` — 以下の手順ではこの形式で統一している。

---

## 1. 背景・概要

### 検証する問題

`cublasSgemm` が `CUBLAS_STATUS_INTERNAL_ERROR` を返してサーバーがクラッシュ（SIGABRT）する問題。  
H2O（Heavy-Hitter Oracle）KV キャッシュ退避が有効な状態で、スロットのコンテキストが別トピックのリクエストで無効化されると
「`no usable hybrid/recurrent checkpoint; forcing full prompt re-processing`」が発生し、
同時に CUBLAS がワークスペース不足でインターナルエラーを起こす。

### 発生条件のサマリー

| 条件 | 値 |
|---|---|
| VRAM | 7850 MiB（RTX 3070）—モデル + KV キャッシュ + compute バッファで残余が数百 MiB |
| KV budget | 100000 トークン（物理 KV は 99840 に丸め込まれる） |
| H2O | 有効（`--kv-budget` + `--kv-sink` + `--kv-evict-interval` の組み合わせ） |
| トリガー操作 | 直前会話と全く別トピックのリクエストが来てチェックポイントが全無効化される |
| 量子化 | Q3_K_XL（低 BPW ＝ VRAM マージンが非常に小さい） |

### 修正内容のサマリー

| 修正 | 内容 |
|---|---|
| `cublasSetWorkspace` 32 MiB 事前確保 | CUBLAS がワークスペースを要求する前に確実にアロケートすることで INTERNAL_ERROR を防ぐ |
| H2O EMA クリア | チェックポイント全無効化時に EMA（Exponential Moving Average）スコアをリセットし、不正な退避判定を防ぐ |

---

## 2. テスト環境要件

### GPU VRAM 要件

本テストは **VRAM 残余の小ささ**がクラッシュのトリガーになるため、以下の構成で再現性が高い。

| 項目 | 実績値 |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 |
| VRAM 合計 | 7850 MiB |
| 起動直後 VRAM 空き | 7681 MiB（OSカーネル分 ~169 MiB 消費） |
| モデル CUDA0 バッファ | 5131.34 MiB |
| KV キャッシュ CUDA0 | 443.67 MiB |
| compute バッファ CUDA0 | 978.00 MiB |
| compute バッファ CUDA_Host | 284.03 MiB |
| pinned host memory | 10923.31 MiB（CPU 側 RAM） |
| **CUDA0 実質残余** | **約 100〜200 MiB**（VRAM フル使用） |

> 8 GB 超の GPU でも、モデルサイズや量子化形式によっては同様の残余になる。  
> VRAM 残余が 500 MiB 以上ある場合はクラッシュが再現しない可能性がある。

### 必要なサーバー設定

以下のオプションがすべて有効であることが前提:

```
--kv-budget 100000   # H2O: KV キャッシュ上限トークン数
--kv-sink 4096       # H2O: Sink トークン数（先頭固定分）
--kv-evict-interval 256  # H2O: 退避チェック間隔
--kv-snapshot-max-mem 8192  # コンテキストスナップショット最大 RAM (MiB)
-ctk turbo3c -ctv turbo3c   # KV キャッシュ量子化形式
--flash-attn 1       # Flash Attention 有効
-c 204800            # コンテキスト長
```

### モデル要件

| 項目 | 値 |
|---|---|
| モデル | Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf |
| エイリアス | qwen3.6-35b-a3b |
| 量子化 | Q3_K_XL（3.886 BPW） |
| サイズ | 15.678 GiB |
| アーキテクチャ | qwen35moe (MoE 40 層、256 エキスパート) |

Q3 系の低ビット量子化は VRAM 節約の一方で **CUBLAS が混合精度演算のためにワークスペースを多く要求する**。  
Q4 以上では同じ操作でもワークスペース要求量が異なる場合がある。

---

## 3. テストチェックリスト

### Pre-conditions チェック

- [ ] サーバーが起動していて `/health` が `{"status":"ok"}` を返す

  ```bash
  ssh vllm "curl -s http://localhost:8080/health"
  # 期待値: {"status":"ok"}
  ```

- [ ] VRAM 空き残量の確認（初期値を記録）

  ```bash
  ssh vllm "nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits"
  # 例: 7630, 220, 7850  (used MiB, free MiB, total MiB)
  # → free MiB を記録しておく
  ```

- [ ] H2O が有効になっていることの確認（起動ログ）

  ```bash
  ssh vllm "grep 'H2O kv_budget' ~/llamacpp/crash_$(ls -t ~/llamacpp/crash_*.log | head -1 | xargs basename).log 2>/dev/null || grep 'H2O kv_budget' ~/llamacpp/$(ls -t ~/llamacpp/*.log 2>/dev/null | head -1)"
  # 期待値: llama_init_from_model: H2O kv_budget=100000, KV cache physical size reduced to 99840 (n_ctx=204800)
  ```

  または最新ログをダイレクトに確認:

  ```bash
  ssh vllm "grep 'H2O kv_budget' ~/llamacpp/crash_$(date +%Y%m%d)_*.log | tail -1"
  ```

- [ ] KV スナップショットストアが有効であることの確認

  ```bash
  ssh vllm "grep 'kv snapshot store enabled' ~/llamacpp/crash_$(date +%Y%m%d)_*.log | tail -1"
  # 期待値: INFO [...] kv snapshot store enabled | ... max_mem_mib=8192
  ```

---

### テスト手順

#### Step 1: ウォームアップリクエスト（トピック A）

同一トピックで複数回リクエストを投げてチェックポイントを積み上げる。  
これにより slot にキャッシュが蓄積される。

```bash
# リクエスト 1-1（ウォームアップ）
ssh vllm "curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer lm-secret-20240427' \
  -d '{
    \"model\": \"qwen3.6-35b-a3b\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Pythonでフィボナッチ数列を計算する関数を書いてください。再帰と反復の両方の実装を示してください。\"}
    ],
    \"max_tokens\": 500,
    \"stream\": false
  }'" | jq '{status: .choices[0].finish_reason, tokens: .usage}'
```

```bash
# リクエスト 1-2（同一トピック継続、チェックポイント追加）
ssh vllm "curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer lm-secret-20240427' \
  -d '{
    \"model\": \"qwen3.6-35b-a3b\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Pythonでフィボナッチ数列を計算する関数を書いてください。再帰と反復の両方の実装を示してください。\"},
      {\"role\": \"assistant\", \"content\": \"はい、Pythonでのフィボナッチ数列の実装を示します。\"},
      {\"role\": \"user\", \"content\": \"その実装の時間計算量をO記法で説明してください。\"}
    ],
    \"max_tokens\": 500,
    \"stream\": false
  }'" | jq '{status: .choices[0].finish_reason, tokens: .usage}'
```

**確認事項:**
- [ ] HTTP 200 が返る
- [ ] ログに `slot create_check: ... created context checkpoint` が出力されている

  ```bash
  ssh vllm "tail -20 ~/llamacpp/crash_$(date +%Y%m%d)_*.log | grep 'create_check'"
  ```

---

#### Step 2: 別トピックリクエスト（チェックポイント無効化テスト）

全く異なるトピックで短いプロンプトを送り、`no usable checkpoint` を意図的に誘発する。

```bash
# リクエスト 2（クラッシュトリガー候補 — 全く別トピック）
ssh vllm "curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer lm-secret-20240427' \
  -d '{
    \"model\": \"qwen3.6-35b-a3b\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"今日の天気はどうですか？\"}
    ],
    \"max_tokens\": 500,
    \"stream\": false
  }'" | jq '{status: .choices[0].finish_reason, tokens: .usage}'
```

**確認事項:**
- [ ] HTTP 200 が返る（クラッシュしていないこと）
- [ ] ログに `no usable hybrid/recurrent checkpoint; forcing full prompt re-processing` が出力される

  ```bash
  ssh vllm "tail -30 ~/llamacpp/crash_$(date +%Y%m%d)_*.log | grep 'no usable'"
  # 期待値（正常）: no usable hybrid/recurrent checkpoint; forcing full prompt re-processing
  # ← このメッセージが出てもサーバーが継続動作していれば修正が効いている
  ```

---

#### Step 3: 連続異なるトピックリクエスト（ストレステスト）

連続して全く異なるトピックを送信し、繰り返し `no usable checkpoint` を誘発し続ける。  
これが修正前のクラッシュの主要トリガーパターン。

```bash
# リクエスト 3-1
ssh vllm "curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer lm-secret-20240427' \
  -d '{\"model\": \"qwen3.6-35b-a3b\", \"messages\": [{\"role\": \"user\", \"content\": \"日本の江戸時代の歴史について教えてください。\"}], \"max_tokens\": 500, \"stream\": false}'" \
  | jq -r '"Request 3-1: " + .choices[0].finish_reason'

# リクエスト 3-2（全く別トピック）
ssh vllm "curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer lm-secret-20240427' \
  -d '{\"model\": \"qwen3.6-35b-a3b\", \"messages\": [{\"role\": \"user\", \"content\": \"量子コンピュータの仕組みを中学生向けに説明してください。\"}], \"max_tokens\": 500, \"stream\": false}'" \
  | jq -r '"Request 3-2: " + .choices[0].finish_reason'

# リクエスト 3-3（全く別トピック）
ssh vllm "curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer lm-secret-20240427' \
  -d '{\"model\": \"qwen3.6-35b-a3b\", \"messages\": [{\"role\": \"user\", \"content\": \"フランス料理とイタリア料理の違いを説明してください。\"}], \"max_tokens\": 500, \"stream\": false}'" \
  | jq -r '"Request 3-3: " + .choices[0].finish_reason'
```

**確認事項:**
- [ ] 全リクエストが `finish_reason: stop` または `length` で返る（クラッシュなし）
- [ ] ログに CUDA error / SIGABRT が出ない

---

#### Step 4: サーバー生存確認

```bash
# Step 3 完了後のヘルスチェック
ssh vllm "curl -s http://localhost:8080/health"
# 期待値: {"status":"ok"}

# プロセス確認
ssh vllm "pgrep -a llama-server"
# プロセスが存在すること
```

---

#### Step 5: VRAM 崩壊チェック

```bash
ssh vllm "nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits"
# 期待値: Pre-conditions で記録した free MiB と大きく変わらないこと
# 目安: 事前値から ±300 MiB 以内
# 異常値: free が 0〜50 MiB 以下（VRAM リーク）
```

---

### 成功判定チェック

- [ ] Step 1〜3 の全リクエストが HTTP 200 で返る
- [ ] `no usable hybrid/recurrent checkpoint; forcing full prompt re-processing` がログに出ても継続動作する
- [ ] ログに `CUDA error` / `CUBLAS_STATUS_INTERNAL_ERROR` / `SIGABRT` が出ない

  ```bash
  ssh vllm "grep -E 'CUDA error|CUBLAS_STATUS|SIGABRT|Aborted|Segmentation fault' ~/llamacpp/crash_$(date +%Y%m%d)_*.log"
  # 出力が空であること
  ```

- [ ] Step 4 のヘルスチェックが `{"status":"ok"}` を返す
- [ ] Step 5 の VRAM 空き残量が崩壊していない（事前値の ±300 MiB 以内、最低 100 MiB 以上残存）

---

## 4. 失敗時の診断

### クラッシュログの場所と確認コマンド

ログは起動スクリプト `run_debug.sh` により自動的に以下に保存される:

```
~/llamacpp/crash_YYYYMMDD_HHMMSS.log
```

最新ログの確認:

```bash
ssh vllm "ls -lt ~/llamacpp/crash_*.log | head -5"
```

GDB バックトレースの確認（`run_debug.sh` は gdb でラップして実行するため、クラッシュ時に bt が記録される）:

```bash
ssh vllm "grep -A 30 'thread apply all bt\|Program received signal\|SIGABRT' ~/llamacpp/crash_$(date +%Y%m%d)_*.log | head -60"
```

### `CUBLAS_STATUS_INTERNAL_ERROR` が出た場合の意味

```
CUBLAS_STATUS_INTERNAL_ERROR (11)
```

CUBLAS ライブラリ内部でのエラー。主な原因:

1. **ワークスペース不足**: `cublasSgemm` 実行時に必要な GPU ワークスペースが確保できない
2. **メモリ断片化**: VRAM がほぼ満杯で連続した空きブロックが取れない
3. **EMA スコア不正**: H2O の退避判定が誤って重要な KV を削除し、後続演算が不正な状態で走る

修正（`cublasSetWorkspace` 32 MiB 事前確保）が適用されていれば、このエラーは発生しない。  
もし再発した場合は 32 MiB が不十分な可能性がある（後述の Section 5 参照）。

### `pool_size` 増加の確認方法

`--kv-snapshot-max-mem` に関連するスナップショットプールサイズの確認:

```bash
ssh vllm "grep 'cache state:\|pool_size\|snapshot saved\|store count' ~/llamacpp/crash_$(date +%Y%m%d)_*.log | tail -20"
# 例:
# slot release_slot: id  0 | task 3028 | snapshot saved: 598 tokens, store count=11, mem=708.2 MiB
# - cache state: 5 prompts, 1267.756 MiB (limits: 8192.000 MiB, ...)
```

`mem=` 値が `--kv-snapshot-max-mem`（8192 MiB）に近づいている場合は上限に接近している。  
`store count` が増加し続ける場合はスナップショット退避が機能していない可能性がある。

---

## 5. モデル変更時の注意事項

### 量子化形式が変わった場合（Q3→Q4 など）の影響

| 変更 | 影響 |
|---|---|
| Q3_K_XL → Q4_K_M | CUDA0 バッファが増加（要実測: 5131 MiB より増加する）。VRAM 残余がさらに小さくなるためクラッシュ再現性が上がる可能性あり |
| Q3_K_XL → Q2_K | CUDA0 バッファが減少。VRAM マージンが増えクラッシュしにくくなるが、精度劣化 |
| BPW が変わる | `llm_load_print_meta: model size` 行を確認し、CUDA0 バッファ + KV バッファ + compute バッファの合計が VRAM 合計を超えないことを確認 |

モデル変更後の VRAM 使用量確認コマンド:

```bash
ssh vllm "grep -E 'CUDA0 buffer size|KV buffer size|compute buffer size|VRAM' ~/llamacpp/crash_$(date +%Y%m%d)_*.log | head -10"
```

### コンテキスト長変更時の VRAM マージン再計算

KV キャッシュサイズは `-c`（コンテキスト長）と量子化形式 `-ctk`/`-ctv` に比例する。

現在の設定での実績値:
- `-c 204800` + `-ctk turbo3c -ctv turbo3c` → KV CUDA0: **443.67 MiB**

コンテキスト長を変更した場合の概算:
```
新 KV MiB ≈ 443.67 × (新コンテキスト長 / 204800)
```

コンテキスト長を増やした場合、VRAM 残余が減る。  
残余が 200 MiB を下回る設定では `cublasSetWorkspace` の 32 MiB 確保自体が失敗するリスクがある。

### `cublasSetWorkspace` の 32 MiB が十分かどうかの判断基準

32 MiB の確保が十分かどうかは以下で判断する:

1. **クラッシュが解消されている**: `CUBLAS_STATUS_INTERNAL_ERROR` が消えていれば 32 MiB で十分
2. **モデルサイズ・量子化が変わった場合**: CUBLAS のワークスペース要求量はバッチサイズ（`-b 1024 -ub 1024`）と行列次元に依存する。バッチサイズを増やした場合（例: 2048）は 64 MiB への増量を検討
3. **確認方法**: クラッシュ直前のスタックトレースに `cublasSgemm` が見える場合は依然としてワークスペース問題。`cublasSetWorkspace` 呼び出し自体が失敗している場合は VRAM 残余の根本不足

再発時の暫定対処（ワークスペースサイズを 64 MiB に拡大する場合）:
```bash
# build.sh を使ってリビルド時にコードを変更
# ggml/src/ggml-cuda.cu 内の cublasSetWorkspace 呼び出しの
# 第3引数を 32*1024*1024 → 64*1024*1024 に変更してリビルド
ssh vllm "cd ~/llamacpp && bash build.sh"
```

---

## 付録: サーバー起動コマンド（参考）

```bash
# run_debug.sh より（gdb ラッパーで起動）
cd ~/llamacpp
gdb -batch \
    -ex "set pagination off" \
    -ex "set print thread-events off" \
    -ex "handle SIGPIPE nostop noprint pass" \
    -ex "run" \
    -ex "thread apply all bt full" \
    -ex "quit" \
    --args env GGML_CUDA_FORCE_MMQ=1 \
        ./ik_llama.cpp/build/bin/llama-server \
        -m models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf \
        -ctk turbo3c -ctv turbo3c \
        -ngl 99 \
        -b 1024 -ub 1024 \
        --flash-attn 1 \
        -t 12 \
        --kv-budget 100000 --kv-sink 4096 --kv-evict-interval 256 \
        --kv-snapshot-max-mem 8192 \
        -c 204800 \
        -tb 24 \
        --n-cpu-moe 31 \
        --mlock \
        --jinja \
        --run-time-repack \
        --alias "qwen3.6-35b-a3b" \
        --api-key lm-secret-20240427 \
        --host 0.0.0.0 \
        --port 8080 \
    2>&1 | tee ~/llamacpp/crash_$(date +%Y%m%d_%H%M%S).log
```

---

## 付録B: チャンク分割変換の検討記録（2026-05-08）

### 検討の背景

起動後に VRAM 空きがじわじわ減少する現象を根治するため、Flash Attention の
K/V 変換バッファ（K_f16 + V_f16）のチャンク分割変換を検討した。

### 調査結果

#### FA バッファの実際のサイズ

K_f16/V_f16 バッファのサイズは `K->ne[1]` = `kv_self.size`（H2O による物理 KV サイズ）に比例する。
H2O `--kv-budget 100000` が有効な場合、`kv_phys = 99840` となり、バッファサイズは：

| 項目 | サイズ |
|---|---|
| K_f16（片方） | ~97.5 MiB |
| V_f16（片方） | ~97.5 MiB |
| **K+V 合計** | **~195 MiB** |

当初想定していた 838 MiB（H2O なし、n_ctx=204800 ベース）とは大きく異なる。
H2O が KV キャッシュ物理サイズを削減することで、FA バッファも自動的に縮小される。

#### K/V バッファの同時確保について

`fattn-mma-f16.cuh` において K_f16 と V_f16 は同一スコープで確保され、
fused な `fattn_kernel` が両方を同時に読むため、sequential 化は構造上不可能。

#### チャンク分割の技術的実現可能性

| 項目 | 内容 |
|---|---|
| アプローチ | `emit_partial` テンプレートフラグ追加 + `flash_attn_mma_combine_results` で合算 |
| 変更ファイル | `fattn-mma-f16.cuh` のみ（約 117 行） |
| decode 時の速度影響 | なし（起動オーバーヘッド < 0.01%） |
| prefill 時 | `dst_tmp` が C×n_queries 分必要になるため C=1 に自動切替が必要 |
| 主なリスク | マスクのチャンクオフセット計算、sink トークンの二重適用防止 |

#### 結論

H2O 有効時の実際の K+V バッファは ~195 MiB であり、
チャンク分割による VRAM 削減効果（195 MiB → ~49 MiB @C=4）は限定的。
実装工数（カーネル改修）に対して効果が小さいため、現時点では優先度低と判断。

VRAM じわじわ減少の真の原因を特定してから再検討する。
