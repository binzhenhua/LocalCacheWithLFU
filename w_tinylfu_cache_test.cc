#include "w_tinylfu_cache.h"

#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define CHECK(cond)                                                      \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, \
                   #cond);                                               \
      return 1;                                                          \
    }                                                                    \
  } while (0)

static std::string RandomBytes(std::mt19937& rng, size_t n) {
  std::string s(n, '\0');
  std::uniform_int_distribution<int> d(0, 255);
  for (auto& c : s) c = static_cast<char>(d(rng));
  return s;
}

static int TestRoundTrip() {
  cache::WTinyLfuCache c(static_cast<size_t>(4) << 20, 1);
  std::mt19937 rng(1);
  std::string v = RandomBytes(rng, 40 * 1024);
  c.Put("k1", v);
  auto got = c.Get("k1");
  CHECK(got.has_value());
  CHECK(*got == v);
  CHECK(!c.Get("missing").has_value());
  CHECK(c.CheckInvariants());
  return 0;
}

static int TestCapacityBound() {
  const size_t cap = static_cast<size_t>(4) << 20;
  cache::WTinyLfuCache c(cap, 1);
  std::mt19937 rng(2);
  for (int i = 0; i < 1000; ++i) {
    c.Put("k" + std::to_string(i), RandomBytes(rng, 40 * 1024));
  }
  auto st = c.Stats();
  CHECK(st.bytes_used <= cap);
  CHECK(st.evictions > 0);
  CHECK(c.CheckInvariants());
  return 0;
}

static int TestUpdateReplacesValue() {
  cache::WTinyLfuCache c(static_cast<size_t>(1) << 20, 1);
  c.Put("a", std::string(100, 'x'));
  c.Put("a", std::string(200, 'y'));
  auto got = c.Get("a");
  CHECK(got.has_value());
  CHECK(*got == std::string(200, 'y'));
  CHECK(c.Stats().entries == 1);
  CHECK(c.CheckInvariants());
  return 0;
}

static int TestOversizedValueRejected() {
  cache::WTinyLfuCache c(static_cast<size_t>(64) << 10, 1);
  c.Put("big", std::string(128 << 10, 'z'));
  CHECK(!c.Get("big").has_value());
  CHECK(c.Stats().rejections >= 1);
  CHECK(c.CheckInvariants());
  return 0;
}

// Frequently accessed keys should survive a one-hit-wonder scan.
static int TestScanResistance() {
  cache::WTinyLfuCache c(static_cast<size_t>(2) << 20, 1);
  std::mt19937 rng(3);
  std::vector<std::string> hot;
  for (int i = 0; i < 10; ++i) {
    hot.push_back("hot" + std::to_string(i));
    c.Put(hot.back(), RandomBytes(rng, 40 * 1024));
  }
  for (int round = 0; round < 20; ++round) {
    for (const auto& k : hot) c.Get(k);
  }
  for (int i = 0; i < 500; ++i) {
    c.Put("scan" + std::to_string(i), RandomBytes(rng, 40 * 1024));
  }
  int hits = 0;
  for (const auto& k : hot) {
    if (c.Get(k).has_value()) ++hits;
  }
  CHECK(hits >= 8);
  CHECK(c.CheckInvariants());
  return 0;
}

static int TestConcurrent() {
  const size_t cap = static_cast<size_t>(8) << 20;
  cache::WTinyLfuCache c(cap, 16);
  auto worker = [&c](int id) {
    std::mt19937 rng(100 + id);
    std::uniform_int_distribution<int> key(0, 499);
    for (int i = 0; i < 2000; ++i) {
      std::string k = "k" + std::to_string(key(rng));
      if (rng() % 2 == 0) {
        c.Put(k, std::string(4096 + (i % 8) * 1024,
                             static_cast<char>('a' + id)));
      } else {
        c.Get(k);
      }
    }
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) threads.emplace_back(worker, i);
  for (auto& t : threads) t.join();
  CHECK(c.CheckInvariants());
  CHECK(c.Stats().bytes_used <= cap);
  return 0;
}

int main() {
  int rc = 0;
  rc |= TestRoundTrip();
  rc |= TestCapacityBound();
  rc |= TestUpdateReplacesValue();
  rc |= TestOversizedValueRejected();
  rc |= TestScanResistance();
  rc |= TestConcurrent();
  if (rc == 0) std::puts("all tests passed");
  return rc;
}