#include "llama-snapshot-store.h"

#include <algorithm>
#include <chrono>
#include <cstring>

std::string kv_snapshot_store::get_model_desc(llama_context * ctx) {
    char buf[256];
    llama_model_desc(llama_get_model(ctx), buf, sizeof(buf));
    return std::string(buf);
}

int64_t kv_snapshot_store::now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

void kv_snapshot_store::set_max_memory(size_t max_bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    max_memory_ = max_bytes;
}

void kv_snapshot_store::save(llama_context * ctx, llama_seq_id seq_id,
                              const std::vector<llama_token> & tokens) {
    if (max_memory_ == 0) {
        return;
    }

    const std::string mdesc = get_model_desc(ctx);

    const size_t kv_size = llama_state_seq_get_size(ctx, seq_id, 0);
    std::vector<uint8_t> kv_data(kv_size);
    const size_t written = llama_state_seq_get_data(ctx, kv_data.data(), kv_size, seq_id, 0);
    if (written == 0) {
        return;
    }
    kv_data.resize(written);

    std::lock_guard<std::mutex> lock(mu_);

    // Replace if new tokens extend (or exactly match) an existing snapshot.
    for (int i = 0; i < (int)snapshots_.size(); ++i) {
        auto & snap = snapshots_[i];
        if (snap.model_desc != mdesc) {
            continue;
        }
        // Check if snap.tokens is a prefix of (or equal to) the new tokens.
        if (tokens.size() >= snap.tokens.size() &&
            std::equal(snap.tokens.begin(), snap.tokens.end(), tokens.begin())) {
            used_memory_ -= snap.kv_data.size();
            snap.tokens      = tokens;
            snap.kv_data     = std::move(kv_data);
            snap.last_access = now_ms();
            // ref_count inherited — do not reset
            used_memory_    += snap.kv_data.size();
            return;
        }
    }

    kv_snapshot snap;
    snap.tokens      = tokens;
    snap.kv_data     = std::move(kv_data);
    snap.model_desc  = mdesc;
    snap.ref_count   = 0;
    snap.last_access = now_ms();

    used_memory_ += snap.kv_data.size();
    snapshots_.push_back(std::move(snap));

    while (used_memory_ > max_memory_ && !snapshots_.empty()) {
        evict_one();
    }
}

int kv_snapshot_store::restore(llama_context * ctx, llama_seq_id seq_id,
                                const std::vector<llama_token> & tokens) {
    if (max_memory_ == 0) {
        return 0;
    }

    std::unique_lock<std::mutex> lock(mu_);

    const std::string mdesc = get_model_desc(ctx);
    const int idx = find_best_prefix(mdesc, tokens);
    if (idx < 0) {
        return 0;
    }

    const std::vector<llama_token> & stok = snapshots_[idx].tokens;
    int prefix_len = 0;
    const int min_len = static_cast<int>(std::min(stok.size(), tokens.size()));
    for (int i = 0; i < min_len; ++i) {
        if (stok[i] == tokens[i]) {
            ++prefix_len;
        } else {
            break;
        }
    }

    snapshots_[idx].ref_count++;
    snapshots_[idx].last_access = now_ms();
    const std::vector<uint8_t> kv_copy = snapshots_[idx].kv_data;

    lock.unlock();

    const size_t read = llama_state_seq_set_data(ctx, kv_copy.data(), kv_copy.size(), seq_id, 0);
    if (read == 0) {
        return 0;
    }

    llama_kv_cache_seq_rm(ctx, seq_id, prefix_len, -1);

    return prefix_len;
}

size_t kv_snapshot_store::total_memory() const {
    std::lock_guard<std::mutex> lock(mu_);
    return used_memory_;
}

int kv_snapshot_store::count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(snapshots_.size());
}

int kv_snapshot_store::find_best_prefix(const std::string & model_desc,
                                         const std::vector<llama_token> & tokens) {
    int best_idx = -1;
    int best_len = 0;

    for (int i = 0; i < static_cast<int>(snapshots_.size()); ++i) {
        const kv_snapshot & snap = snapshots_[i];
        if (snap.model_desc != model_desc) {
            continue;
        }

        const int min_len = static_cast<int>(std::min(snap.tokens.size(), tokens.size()));
        int prefix_len = 0;
        for (int j = 0; j < min_len; ++j) {
            if (snap.tokens[j] == tokens[j]) {
                ++prefix_len;
            } else {
                break;
            }
        }

        if (prefix_len > best_len) {
            best_len = prefix_len;
            best_idx = i;
        }
    }

    return (best_len > 0) ? best_idx : -1;
}

void kv_snapshot_store::evict_one() {
    if (snapshots_.empty()) {
        return;
    }

    int evict_idx = 0;
    for (int i = 1; i < static_cast<int>(snapshots_.size()); ++i) {
        const kv_snapshot & cand    = snapshots_[i];
        const kv_snapshot & current = snapshots_[evict_idx];
        if (cand.ref_count < current.ref_count ||
            (cand.ref_count == current.ref_count && cand.last_access < current.last_access)) {
            evict_idx = i;
        }
    }

    used_memory_ -= snapshots_[evict_idx].kv_data.size();
    snapshots_.erase(snapshots_.begin() + evict_idx);
}
