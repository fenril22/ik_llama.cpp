# ik_llama.cpp プロジェクト CLAUDE.md

## テスト環境

- **vllmサーバー**: SSH接続 `ssh vllm`
  - パス: `/home/vllm/llamacpp/`
  - GPU: NVIDIA RTX 3070 (7,850 MiB VRAM)
  - モデル: `models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf`
  - 起動スクリプト: `bash run_ik.sh`（PATH=/usr/local/cuda/bin:$PATH が必要）
  - ビルドスクリプト: `PATH=/usr/local/cuda/bin:$PATH bash build.sh`
  - バイナリ: `ik_llama.cpp/build/bin/llama-server`、`ik_llama.cpp/build/bin/llama-bench`
- **ローカル**: `/Users/yabunaka/tmp/ik_llama.cpp/`
- **リモート**: `origin/feature/rotorquant-kv-cache`

## ビルド・デプロイ手順

1. ローカルで編集
2. `git add && git commit && git push`
3. `ssh vllm "cd /home/vllm/llamacpp && git pull"` （SSH認証の場合はscpでコピー）
4. `ssh vllm "cd /home/vllm/llamacpp && PATH=/usr/local/cuda/bin:\$PATH bash build.sh"`

## 主要パラメータ（現在の本番設定）

- `-b 2048 -ub 512`: バッチサイズ（ubが小さいほどVRAM節約、512が最適バランス）
- `--n-cpu-moe 28`: MoEレイヤー28層をCPUオフロード（tg ~62.6 t/s）
- `--kv-budget 102400`: H2O KV eviction上限
- `-c 204800`: コンテキスト長
- `-np 1`: スロット数（Agentワークロードではnp=1が最速）
- `-ctk turbo3c -ctv turbo3c`: KV量子化（3-bit、VRAM節約）
- `GGML_CUDA_POOL_VMM_PREALLOC_MiB=200`: VMM事前確保（起動時に設定）

## テスト方法

```bash
# サーバー起動
ssh vllm "cd /home/vllm/llamacpp && nohup bash -c 'export PATH=/usr/local/cuda/bin:\$PATH && bash run_ik.sh' > /tmp/server.log 2>&1 &"

# 動作確認
curl http://vllm:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ik_llm","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'

# VRAMモニタリング
ssh vllm "nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader"

# ベンチマーク（サーバー停止後）
ssh vllm "cd /home/vllm/llamacpp && PATH=/usr/local/cuda/bin:\$PATH ./build/bin/llama-bench \
  -m ik_llama.cpp/models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf \
  -ngl 99 --flash-attn 1 -b 2048 -ub 512 --n-cpu-moe 28 \
  -p 512 -n 128 2>&1 | tail -10"
```

## 注意事項

- vllmサーバーへのgit pullはSSH認証なしのHTTPS経由のみ可能（SSH鍵未設定）
- `llama-bench`では`-ctk turbo3c`がパーサーエラーになるため`q8_0`等で代替
- サーバー起動には`PATH=/usr/local/cuda/bin:$PATH`が必須（nvccのパス）
- Compute Bufferウォームアップにより起動時間が数秒増加（正常動作）
- 起動後ログで`warmup complete`と`pre-allocated 200 MiB`を確認すること

## 主要ファイル

| ファイル | 説明 |
|---|---|
| `src/llama.cpp` | warmupブロック（`llama_init_from_model`末尾） |
| `ggml/src/ggml-cuda.cu` | VMM poolプリアロケーション |
| `ggml/src/ggml-cuda/common.cuh` | cuBLASワークスペース確保 |
| `src/llama-h2o.cpp` | H2O KV eviction実装 |
| `examples/server/server-context.cpp` | サーバースロット管理・checkpoint |
| `docs/serving-optimization.md` | このプロジェクトの最適化知見まとめ |
| `docs/vram-investigation.md` | VRAM消費挙動の詳細調査 |
| `docs/test-cublas-crash-scenario.md` | cublasSgemmクラッシュテスト手順 |
