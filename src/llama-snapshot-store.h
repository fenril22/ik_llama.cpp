#pragma once

#include "llama.h"
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct kv_snapshot {
    std::vector<llama_token> tokens;
    std::vector<uint8_t>     kv_data;
    std::string              model_desc; // for validation
    uint32_t                 ref_count = 0;
    int64_t                  last_access = 0;
};

class kv_snapshot_store {
public:
    // max_bytes == 0 means disabled (no saves)
    void set_max_memory(size_t max_bytes);
    size_t get_max_memory() const { return max_memory_; }

    // Save current KV state of seq_id with its token list.
    // Overwrites existing entry with same model_desc if tokens fully match.
    void save(llama_context * ctx, llama_seq_id seq_id,
              const std::vector<llama_token> & tokens);

    // Find best prefix match, restore into seq_id, trim excess KV.
    // Returns number of tokens restored (0 = miss or disabled).
    int restore(llama_context * ctx, llama_seq_id seq_id,
                const std::vector<llama_token> & tokens);

    size_t total_memory() const;
    int    count() const;

private:
    mutable std::mutex       mu_;
    std::vector<kv_snapshot> snapshots_;
    size_t max_memory_ = 0;
    size_t used_memory_ = 0;

    // Returns index of best prefix match, or -1. Must be called with mu_ held.
    int find_best_prefix(const std::string & model_desc,
                         const std::vector<llama_token> & tokens);
    // Evict lowest ref_count (tiebreak: oldest last_access). mu_ must be held.
    void evict_one();

    static std::string get_model_desc(llama_context * ctx);
    static int64_t     now_ms();
};
