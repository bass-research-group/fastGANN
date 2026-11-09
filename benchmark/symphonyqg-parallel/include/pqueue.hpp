#pragma once

#include <vector>
#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <cfloat>

template <typename T>
class new_pqueue_t {
private:
  int queue_size;
  int queue_capacity;
  // maintain invariant that next_idx and next_idx2 are always the two smallest
  // unexpanded positions. If they don't exist, then they equal queue_size.
  int next_idx, next_idx2;
  std::vector<T> vid_queue;
  std::vector<float> distances;
  std::vector<uint8_t> expanded;

public:
  new_pqueue_t(int L) {
    next_idx = next_idx2 = 0;
    queue_size = 0;
    queue_capacity = L;
    vid_queue.resize(L+1);
    distances.resize(L+1);
    expanded.resize(L+1);
    std::fill(expanded.begin(), expanded.end(), 0);
  }

  T& operator[](size_t i) { return vid_queue[i]; }
  const T& operator[](size_t i) const { return vid_queue[i]; }
  int get_next_index() const { return next_idx; }
  float get_tail_dist() const { return queue_size == 0 ? FLT_MAX : distances[queue_size-1]; }
  int size() { return queue_size; }

  bool has_unexpanded() { return next_idx < queue_size; }
  void set_front_expanded() {
    expanded[next_idx] = 1;
    next_idx = next_idx2;
    if (next_idx2 < queue_size) next_idx2++;
    while ((next_idx2 < queue_size) && expanded[next_idx2]) next_idx2++;
  }
  void push(T vid, float dist) {
    if ((queue_size == queue_capacity) && (dist > distances[queue_size-1]))
      return; // queue is full
    auto queue_start = &distances[0];
    const auto loc = std::lower_bound(queue_start, queue_start + queue_size, dist);
    auto insert_loc = loc - queue_start;
    if (insert_loc < queue_size) {
      auto num = queue_size - insert_loc;
      memmove(reinterpret_cast<char *>(&vid_queue[insert_loc + 1]),
              reinterpret_cast<char *>(&vid_queue[insert_loc]),
              num * sizeof(T));
      memmove(reinterpret_cast<char *>(&distances[insert_loc + 1]),
              reinterpret_cast<char *>(&distances[insert_loc]),
              num * sizeof(float));
      memmove(reinterpret_cast<char *>(&expanded[insert_loc + 1]),
              reinterpret_cast<char *>(&expanded[insert_loc]),
              num * sizeof(uint8_t));
    }
    vid_queue[insert_loc] = vid;
    distances[insert_loc] = dist;
    expanded[insert_loc] = 0;
    if (queue_size < queue_capacity) queue_size++;
    if (insert_loc <= next_idx) {
      next_idx2 = (next_idx == queue_size) ? queue_size:next_idx+1;
      next_idx = insert_loc;
    }
    else if (insert_loc <= next_idx2) next_idx2 = insert_loc;
  }

};
