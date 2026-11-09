#pragma once

#include <vector>
#include <unordered_set>
#include <stdint.h>
#include <string.h>
#include <cfloat>
#include <limits>

template <typename T>
class hash_filter {
private:
  int bits;
  int capacity;
  std::vector<T> seen;
  std::unordered_set<T> stl_hash_;
  static constexpr T kEmptyValue = std::numeric_limits<T>::max();

public:
  hash_filter(int L) {
    int bit_size = 0;
    int tmp = L;
    while (tmp != 0) { bit_size++; tmp >>= 1; }
    bits = (bit_size + 4) / 2 + 3;
    capacity = 1 << bits;
    seen.resize(capacity);
    fill(seen.begin(), seen.end(), kEmptyValue);
  }
  int get_hash(T thing) {
    return thing & ((1 << bits)-1);
  }

  bool contains(T thing) {
    T val = seen[get_hash(thing)];
    if (val == thing) return true;
    if (val == kEmptyValue) return false;
    return stl_hash_.find(thing) != stl_hash_.end();
  }
  bool add(T thing) {
    T& val = seen[get_hash(thing)];
    if (val == thing) return false;
    if (val == kEmptyValue) { val = thing; return true; }
    stl_hash_.insert(thing);
    return true;
  }

};
