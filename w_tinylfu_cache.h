#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "snappy.h"

namespace cache {

struct CacheStats {
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t puts = 0;
  uint64_t evictions = 0;
  uint64_t rejections = 0;  // admission rejects + oversized values
  uint64_t bytes_used = 0;  // key bytes + compressed value bytes
  uint64_t entries = 0;
};

// ---------- Hashing ----------
// Murmur-style mixing. For production, consider wyhash or xxh3.
inline uint64_t Mix64(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

inline uint64_t Hash64(std::string_view s) {
  uint64_t h = 0x9E3779B97F4A7C15ULL ^ (s.size() * 0xC2B2AE3D27D4EB4FULL);
  size_t i = 0;
  for (; i + 8 <= s.size(); i += 8) {
    uint64_t k;
    std::memcpy(&k, s.data() + i, 8);
    h ^= Mix64(k);
    h = ((h << 27) | (h >> 37)) * 0x94D049BB133111EBULL + 0x27D4EB2F165667C5ULL;
  }
  if (i < s.size()) {
    uint64_t k = 0;
    std::memcpy(&k, s.data() + i, s.size() - i);
    h ^= Mix64(k);
  }
  return Mix64(h);
}

// ---------- Count-Min Sketch with periodic halving (aging) ----------
// Each counter is capped at max_count. The default of 15 fits in 4 bits,
// which is the usual TinyLFU choice. Raise it only if you have measured
// that frequencies above 15 matter for admission decisions.
class CountMinSketch {
 public:
  CountMinSketch(size_t width, size_t depth, uint64_t sample_size,
                 uint8_t max_count)
      : width_(RoundUpPow2(width)),
        depth_(depth),
        sample_size_(sample_size),
        max_count_(max_count),
        counters_(depth * width_, 0) {}

  void Increment(uint64_t hash) {
    for (size_t r = 0; r < depth_; ++r) {
      uint8_t& c = counters_[Index(r, hash)];
      if (c < max_count_) ++c;
    }
    if (++additions_ >= sample_size_) Halve();
  }

  uint8_t Estimate(uint64_t hash) const {
    uint8_t m = max_count_;
    for (size_t r = 0; r < depth_; ++r) {
      m = std::min(m, counters_[Index(r, hash)]);
    }
    return m;
  }

 private:
  static size_t RoundUpPow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
  }

  size_t Index(size_t row, uint64_t hash) const {
    uint64_t h = Mix64(hash + row * 0x9E3779B97F4A7C15ULL);
    return row * width_ + (h & (width_ - 1));
  }

  void Halve() {
    for (auto& c : counters_) c >>= 1;
    additions_ = 0;
  }

  size_t width_;
  size_t depth_;
  uint64_t sample_size_;
  uint8_t max_count_;
  uint64_t additions_ = 0;
  std::vector<uint8_t> counters_;
};

// ---------- Entries ----------
enum class Segment : uint8_t { kWindow = 0, kProbation = 1, kProtected = 2 };
inline constexpr int kNumSegments = 3;

// One heap allocation holds both key and compressed value.
struct Node {
  std::unique_ptr<char[]> buf;  // [key bytes][compressed value bytes]
  size_t key_len = 0;
  size_t val_len = 0;
  uint64_t hash = 0;
  Segment seg = Segment::kWindow;

  std::string_view key() const { return {buf.get(), key_len}; }
  std::string_view value() const { return {buf.get() + key_len, val_len}; }
  size_t charge() const { return key_len + val_len; }
};

inline std::unique_ptr<char[]> MakeBuffer(std::string_view key,
                                          std::string_view val) {
  auto buf = std::make_unique<char[]>(key.size() + val.size());
  std::memcpy(buf.get(), key.data(), key.size());
  std::memcpy(buf.get() + key.size(), val.data(), val.size());
  return buf;
}

// ---------- One shard of W-TinyLFU ----------
// Layout: window (~1%), probation, protected (~80% of capacity, SLRU).
class Shard {
 public:
  Shard(size_t capacity_bytes, size_t sketch_width, uint64_t sketch_sample)
      : capacity_(capacity_bytes),
        window_cap_(capacity_bytes / 100),
        protected_cap_(capacity_bytes / 100 * 80),
        sketch_(sketch_width, /*depth=*/4, sketch_sample, /*max_count=*/15) {}

  std::optional<std::string> Get(std::string_view key, uint64_t h) {
    std::string compressed;
    {
      std::lock_guard<std::mutex> lock(mu_);
      sketch_.Increment(h);
      auto mi = index_.find(key);
      if (mi == index_.end()) {
        ++misses_;
        return std::nullopt;
      }
      ++hits_;
      auto it = mi->second;
      auto v = it->value();
      compressed.assign(v.data(), v.size());
      Touch(it);
    }
    // Decompress outside the lock.
    size_t len = 0;
    if (!snappy::GetUncompressedLength(compressed.data(), compressed.size(),
                                       &len)) {
      return std::nullopt;
    }
    std::string out(len, '\0');
    if (!snappy::RawUncompress(compressed.data(), compressed.size(),
                               out.data())) {
      return std::nullopt;
    }
    return out;
  }

  void Put(std::string_view key, uint64_t h, std::string_view value) {
    std::string compressed;
    snappy::Compress(value.data(), value.size(), &compressed);
    const size_t charge = key.size() + compressed.size();

    std::lock_guard<std::mutex> lock(mu_);
    sketch_.Increment(h);
    ++puts_;

    auto mi = index_.find(key);
    if (charge > capacity_) {
      // Too large to ever fit. Drop any stale entry for this key.
      ++rejections_;
      if (mi != index_.end()) Remove(mi->second);
      return;
    }

    if (mi != index_.end()) {
      // Update in place. Erase the index entry first, because the old key
      // view points into the buffer that is about to be replaced.
      auto it = mi->second;
      Node& n = *it;
      Detach(n);
      index_.erase(mi);
      n.buf = MakeBuffer(key, compressed);
      n.key_len = key.size();
      n.val_len = compressed.size();
      n.hash = h;
      Attach(n);
      index_.emplace(n.key(), it);
      Touch(it);
    } else {
      Node node;
      node.buf = MakeBuffer(key, compressed);
      node.key_len = key.size();
      node.val_len = compressed.size();
      node.hash = h;
      node.seg = Segment::kWindow;

      List& w = ListFor(Segment::kWindow);
      w.push_front(std::move(node));
      auto it = w.begin();
      Attach(*it);
      index_.emplace(it->key(), it);
    }
    Evict();
  }

  CacheStats Stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    CacheStats s;
    s.hits = hits_;
    s.misses = misses_;
    s.puts = puts_;
    s.evictions = evictions_;
    s.rejections = rejections_;
    s.bytes_used = total_;
    s.entries = index_.size();
    return s;
  }

  // Verifies that the incremental counters match the actual contents.
  bool CheckInvariants() const {
    std::lock_guard<std::mutex> lock(mu_);
    size_t sum = 0;
    size_t nodes = 0;
    for (int s = 0; s < kNumSegments; ++s) {
      size_t seg_sum = 0;
      for (const Node& n : lists_[s]) {
        if (static_cast<int>(n.seg) != s) return false;
        seg_sum += n.charge();
        ++nodes;
      }
      if (seg_sum != seg_bytes_[s]) return false;
      sum += seg_sum;
    }
    return sum == total_ && nodes == index_.size() && total_ <= capacity_;
  }

 private:
  using List = std::list<Node>;

  List& ListFor(Segment s) { return lists_[static_cast<int>(s)]; }

  void Attach(const Node& n) {
    seg_bytes_[static_cast<int>(n.seg)] += n.charge();
    total_ += n.charge();
  }

  void Detach(const Node& n) {
    seg_bytes_[static_cast<int>(n.seg)] -= n.charge();
    total_ -= n.charge();
  }

  // Moves a node to the front of another segment. Total bytes do not change.
  void MoveTo(List::iterator it, Segment dst) {
    const size_t c = it->charge();
    List& from = ListFor(it->seg);
    List& to = ListFor(dst);
    seg_bytes_[static_cast<int>(it->seg)] -= c;
    seg_bytes_[static_cast<int>(dst)] += c;
    it->seg = dst;
    to.splice(to.begin(), from, it);
  }

  void Touch(List::iterator it) {
    switch (it->seg) {
      case Segment::kWindow:
        ListFor(Segment::kWindow)
            .splice(ListFor(Segment::kWindow).begin(),
                    ListFor(Segment::kWindow), it);
        break;
      case Segment::kProbation:
        MoveTo(it, Segment::kProtected);
        RebalanceProtected();
        break;
      case Segment::kProtected:
        ListFor(Segment::kProtected)
            .splice(ListFor(Segment::kProtected).begin(),
                    ListFor(Segment::kProtected), it);
        break;
    }
  }

  // Demote the coldest protected entries back to probation when over budget.
  void RebalanceProtected() {
    List& prot = ListFor(Segment::kProtected);
    while (seg_bytes_[static_cast<int>(Segment::kProtected)] > protected_cap_ &&
           !prot.empty()) {
      MoveTo(std::prev(prot.end()), Segment::kProbation);
    }
  }

  void Remove(List::iterator it) {
    Detach(*it);
    index_.erase(it->key());
    ListFor(it->seg).erase(it);
    ++evictions_;
  }

  // Admission: a window candidate must beat the probation victim on frequency
  // to enter the main cache. Otherwise it is rejected.
  void Evict() {
    List& win = ListFor(Segment::kWindow);
    List& prob = ListFor(Segment::kProbation);
    List& prot = ListFor(Segment::kProtected);

    while (total_ > capacity_) {
      const bool window_over =
          seg_bytes_[static_cast<int>(Segment::kWindow)] > window_cap_;
      const bool main_empty = prob.empty() && prot.empty();

      if (!win.empty() && (window_over || main_empty)) {
        auto cand = std::prev(win.end());
        if (prob.empty()) {
          MoveTo(cand, Segment::kProbation);  // nothing to compete with
          continue;
        }
        auto victim = std::prev(prob.end());
        if (sketch_.Estimate(cand->hash) > sketch_.Estimate(victim->hash)) {
          Remove(victim);
          MoveTo(cand, Segment::kProbation);
        } else {
          ++rejections_;
          Remove(cand);
        }
      } else if (!prob.empty()) {
        Remove(std::prev(prob.end()));
      } else if (!prot.empty()) {
        Remove(std::prev(prot.end()));
      } else if (!win.empty()) {
        Remove(std::prev(win.end()));
      } else {
        break;
      }
    }
  }

  mutable std::mutex mu_;
  size_t capacity_;
  size_t window_cap_;
  size_t protected_cap_;

  size_t total_ = 0;
  size_t seg_bytes_[kNumSegments] = {0, 0, 0};

  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
  uint64_t puts_ = 0;
  uint64_t evictions_ = 0;
  uint64_t rejections_ = 0;

  CountMinSketch sketch_;
  List lists_[kNumSegments];
  absl::flat_hash_map<std::string_view, List::iterator> index_;
};

// ---------- Sharded front-end ----------
class WTinyLfuCache {
 public:
  WTinyLfuCache(size_t total_bytes, size_t num_shards = 16,
                size_t sketch_width = size_t{1} << 18) {
    const size_t per_shard = total_bytes / num_shards;
    shards_.reserve(num_shards);
    for (size_t i = 0; i < num_shards; ++i) {
      shards_.push_back(std::make_unique<Shard>(per_shard, sketch_width,
                                                sketch_width * 10));
    }
  }

  std::optional<std::string> Get(std::string_view key) {
    const uint64_t h = Hash64(key);
    return ShardFor(h).Get(key, h);
  }

  void Put(std::string_view key, std::string_view value) {
    const uint64_t h = Hash64(key);
    ShardFor(h).Put(key, h, value);
  }

  CacheStats Stats() const {
    CacheStats total;
    for (const auto& s : shards_) {
      CacheStats t = s->Stats();
      total.hits += t.hits;
      total.misses += t.misses;
      total.puts += t.puts;
      total.evictions += t.evictions;
      total.rejections += t.rejections;
      total.bytes_used += t.bytes_used;
      total.entries += t.entries;
    }
    return total;
  }

  bool CheckInvariants() const {
    for (const auto& s : shards_) {
      if (!s->CheckInvariants()) return false;
    }
    return true;
  }

 private:
  // Use high bits for shard selection, so it is not correlated with the
  // low bits the sketch uses for its row indexes.
  Shard& ShardFor(uint64_t h) {
    return *shards_[(h >> 32) % shards_.size()];
  }

  std::vector<std::unique_ptr<Shard>> shards_;
};

}  // namespace cache