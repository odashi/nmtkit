#include "config.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <nmtkit/exception.h>
#include <nmtkit/huffman_code.h>

using std::function;
using std::max;
using std::priority_queue;
using std::vector;

namespace nmtkit {

HuffmanCode::HuffmanCode(const Vocabulary & vocab) {
  // Comparator of two nodes.
  auto cmpnode = [&](unsigned a, unsigned b) -> bool {
    return pool_[a].freq > pool_[b].freq;
  };

  // Stores leaf nodes.
  // Make initial priority queue.
  priority_queue<unsigned, vector<unsigned>, decltype(cmpnode)> pq(cmpnode);
  for (unsigned i = 0; i < vocab.size(); ++i) {
    pool_.emplace_back(Node {vocab.getFrequency(i), 0, -1, -1, {}});
    pq.push(i);
  }

  // Constructs Huffman tree.
  while (pq.size() > 1) {
    const int l = pq.top();
    pq.pop();
    const int r = pq.top();
    pq.pop();
    const unsigned freq = pool_[l].freq + pool_[r].freq;
    const unsigned depth = max(pool_[l].depth, pool_[r].depth) + 1;
    pool_.emplace_back(Node {freq, depth, l, r, {}});
    pq.push(pool_.size() - 1);
  }

  // NOTE: After above processes, the size of pool_ becomes:
  //       2 * vocab.size() - 1.
  //       And the last element of pool_ becomes the root node of the Huffman
  //       tree.

  // Calculates binary code for each word.
  function<void(unsigned, unsigned, vector<bool> &)> traverse = [&](
      unsigned cur, unsigned depth, vector<bool> & code) {
    Node & node = pool_[cur];
    if (node.left == -1) {
      // Leaf
      node.code = code;
    } else {
      // Branch
      code[depth] = true;
      traverse(node.right, depth + 1, code);
      code[depth] = false;
      traverse(node.left, depth + 1, code);
    }
  };
  vector<bool> code(pool_.back().depth);
  traverse(pool_.size() - 1, 0, code);
}

vector<bool> HuffmanCode::getCode(const unsigned id) const {
  NMTKIT_CHECK(id <= pool_.size() / 2, "id should be less than vocab_size.");
  return pool_[id].code;
}

unsigned HuffmanCode::getID(const vector<bool> & code) const {
  NMTKIT_CHECK_EQ(pool_.back().depth, code.size(), "Invalid length of code.");
  unsigned cur = pool_.size() - 1;
  unsigned depth = 0;
  while (pool_[cur].left != -1) {
    cur = code[depth] ? pool_[cur].right : pool_[cur].left;
    ++depth;
  }
  return cur;
}

unsigned HuffmanCode::getNumBits() const {
  return pool_.back().depth;
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::HuffmanCode);
