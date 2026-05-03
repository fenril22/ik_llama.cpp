#!/bin/bash
# bench-turbo3c.sh — Benchmark turbo3c KV cache configurations
#
# Usage:
#   ./scripts/bench-turbo3c.sh [model_path] [text_file]
#
# Defaults assume the standard vllm server setup.
# Requires: llama-cli, llama-perplexity built with CUDA

set -e

# ── Configuration ──
MODEL="${1:-models/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated.i1-IQ3_S.gguf}"
TEXT="${2:-t128.txt}"
BIN_DIR="${BIN_DIR:-./ik_llama.cpp/build/bin}"
COMMON_FLAGS="-ngl 99 -b 1024 -ub 1024 --flash-attn 1 -t 12 -tb 24 --n-cpu-moe 30 --mlock --run-time-repack"

# ── KV configurations to test ──
declare -A CONFIGS
CONFIGS["f16"]="-ctk f16 -ctv f16"
CONFIGS["q4_0"]="-ctk q4_0 -ctv q4_0"
CONFIGS["q4_1"]="-ctk q4_1 -ctv q4_1"
CONFIGS["turbo3c"]="-ctk turbo3c -ctv turbo3c"
CONFIGS["t3c+q4_1_FL5"]="-ctk turbo3c -ctv turbo3c -ctk-first q4_1,5 -ctv-first q4_1,5 -ctk-last q4_1,5 -ctv-last q4_1,5"
CONFIGS["t3c+q4_1_FL10"]="-ctk turbo3c -ctv turbo3c -ctk-first q4_1,10 -ctv-first q4_1,10 -ctk-last q4_1,10 -ctv-last q4_1,10"

# ── Functions ──
run_decode_bench() {
    local name="$1"
    local kv_flags="$2"
    local ctx="$3"

    echo -n "  decode @${ctx}k: "
    result=$(timeout 600 ${BIN_DIR}/llama-cli \
        -m "$MODEL" $kv_flags $COMMON_FLAGS \
        -c $((ctx * 1024)) -f "$TEXT" -n 20 2>&1 | grep "eval time" | head -1)

    if [ -n "$result" ]; then
        ms_per_tok=$(echo "$result" | grep -oP '\(\s*\K[0-9.]+(?=\s*ms per token)')
        tok_per_s=$(echo "$result" | grep -oP '[0-9.]+(?=\s*tokens per second)' | tail -1)
        echo "${tok_per_s} tok/s (${ms_per_tok} ms/tok)"
    else
        echo "FAILED or TIMEOUT"
    fi
}

run_ppl_bench() {
    local name="$1"
    local kv_flags="$2"
    local ctx="$3"
    local chunks="$4"

    echo -n "  PPL @${ctx}k (${chunks} chunks): "
    result=$(timeout 600 ${BIN_DIR}/llama-perplexity \
        -m "$MODEL" $kv_flags $COMMON_FLAGS \
        -c $((ctx * 1024)) -f "$TEXT" --chunks "$chunks" 2>&1 | grep "Final estimate")

    if [ -n "$result" ]; then
        ppl=$(echo "$result" | grep -oP 'PPL.*= \K[0-9.]+')
        echo "$ppl"
    else
        echo "FAILED or TIMEOUT"
    fi
}

# ── Main ──
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         TURBO3C KV Cache Benchmark Suite                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║ Model: $(basename $MODEL)"
echo "║ Text:  $TEXT"
echo "║ Date:  $(date '+%Y-%m-%d %H:%M')"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

for name in f16 q4_0 q4_1 turbo3c "t3c+q4_1_FL5" "t3c+q4_1_FL10"; do
    kv_flags="${CONFIGS[$name]}"
    echo "━━━ $name ━━━"
    echo "  flags: $kv_flags"

    # Get KV size from short run
    kv_info=$(timeout 120 ${BIN_DIR}/llama-cli \
        -m "$MODEL" $kv_flags $COMMON_FLAGS \
        -c 4096 -p "Hi" -n 1 2>&1 | grep "KV self size" || echo "")
    if [ -n "$kv_info" ]; then
        echo "  $kv_info"
    fi

    # PPL tests
    run_ppl_bench "$name" "$kv_flags" 8 10
    run_ppl_bench "$name" "$kv_flags" 32 3

    # Decode speed (skip f16 at 128k - likely OOM)
    if [ "$name" != "f16" ]; then
        run_decode_bench "$name" "$kv_flags" 128
    fi

    echo
done

echo "Done."
