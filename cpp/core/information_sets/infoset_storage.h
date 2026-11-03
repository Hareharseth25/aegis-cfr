// Contiguous per-information-set storage.
//
// Every information set I owns the global action slots
// [I.offset, I.offset + |A(I)|). Regrets, strategy sums and policies are flat
// arrays over slots, so all values of one information set are adjacent in
// memory and a whole profile is a single allocation (no object per node).
// These helpers give typed views onto that layout.
#pragma once

#include <span>
#include <vector>

#include "core/game_tree/game_tree.h"
#include "util/check.h"

namespace aegis {

template <typename T>
std::span<T> infoset_span(const GameTree& tree, std::vector<T>& slots, int infoset) {
  AEGIS_CHECK_MSG(infoset >= 0 && infoset < tree.num_infosets(), "infoset " << infoset << " out of range");
  const auto& I = tree.infosets[infoset];
  return {slots.data() + I.offset, static_cast<size_t>(I.num_actions)};
}

template <typename T>
std::span<const T> infoset_span(const GameTree& tree, const std::vector<T>& slots, int infoset) {
  AEGIS_CHECK_MSG(infoset >= 0 && infoset < tree.num_infosets(), "infoset " << infoset << " out of range");
  const auto& I = tree.infosets[infoset];
  return {slots.data() + I.offset, static_cast<size_t>(I.num_actions)};
}

// A slot-indexed array sized for a tree (one value per (infoset, action)).
template <typename T>
class SlotArray {
 public:
  SlotArray() = default;
  explicit SlotArray(const GameTree& tree, T init = T{}) : data_(tree.total_actions, init) {}
  std::span<T> at(const GameTree& tree, int infoset) { return infoset_span(tree, data_, infoset); }
  std::span<const T> at(const GameTree& tree, int infoset) const { return infoset_span(tree, data_, infoset); }
  std::vector<T>& raw() { return data_; }
  const std::vector<T>& raw() const { return data_; }
  size_t size() const { return data_.size(); }

 private:
  std::vector<T> data_;
};

}  // namespace aegis
