#include <nmtkit/inference_graph.h>

#include <algorithm>
#include <set>
#include <nmtkit/array.h>
#include <nmtkit/exception.h>

using std::vector;

namespace nmtkit {

InferenceGraph::~InferenceGraph() {
  clear();
}

void InferenceGraph::clear() {
  while (!nodes_.empty()) {
    Node * node = nodes_.back();
    nodes_.pop_back();
    delete node;
  }
}

InferenceGraph::Node * InferenceGraph::addNode(const Label & label) {
  nodes_.emplace_back(new Node(label));
  return nodes_.back();
}

void InferenceGraph::connect(Node * prev, Node * next) {
  NMTKIT_CHECK(
      find(nodes_.begin(), nodes_.end(), prev) != nodes_.end(),
      "The graph does not include given 'prev' node.");
  NMTKIT_CHECK(
      find(nodes_.begin(), nodes_.end(), next) != nodes_.end(),
      "The graph does not include given 'next' node.");
  vector<const Node *> & pn = prev->next_;
  vector<const Node *> & np = next->prev_;
  if (find(pn.begin(), pn.end(), next) == pn.end()) {
    // Connect prev and next if both are not already connected.
    pn.emplace_back(next);
    np.emplace_back(prev);
  }
}

vector<const InferenceGraph::Node *> InferenceGraph::findNodes(
    std::function<bool(const Node &)> cond) const {
  vector<const Node *> results;
  for (const Node * node : nodes_) {
    if (cond(*node)) {
      results.emplace_back(node);
    }
  }
  return results;
}

vector<const InferenceGraph::Node *> InferenceGraph::findOneBestPath(
    const unsigned bos_id,
    const unsigned eos_id) const {
  // Finds <s> and </s> nodes.
  auto bos_nodes = findNodes([&](const Node & node) {
      return node.label().word_id == bos_id && node.prev().size() == 0;
  });
  auto eos_nodes = findNodes([&](const Node & node) {
      return node.label().word_id == eos_id && node.next().size() == 0;
  });
  NMTKIT_CHECK_NE(
      0, bos_nodes.size(),
      "Detected no <s> nodes in the inference graph.");
  NMTKIT_CHECK_EQ(
      1, bos_nodes.size(),
      "Detected mulriple <s> nodes in the inference graph.");
  NMTKIT_CHECK(
      !eos_nodes.empty(),
      "Detected no </s> nodes in the inference graph.");
  const Node * bos_node = bos_nodes[0];

  // Finds the </s> node which has the largest probability.
  const Node * best_node = nullptr;
  float best_accum_log_prob = -1e10f;
  for (const Node * node : eos_nodes) {
    if (node->label().accum_log_prob > best_accum_log_prob) {
      best_node = node;
      best_accum_log_prob = node->label().accum_log_prob;
    }
  }

  // Traverses the path.
  vector<const Node *> results {best_node};
  std::set<const Node *> visited {best_node};
  while (best_node != bos_node) {
    NMTKIT_CHECK_EQ(
        best_node->prev().size(), 1,
        "Detected current node has no or multiple previous nodes.");
    best_node = best_node->prev()[0];
    NMTKIT_CHECK_EQ(
        visited.find(best_node), visited.end(),
        "Detected a loop in the inference graph.");
    results.emplace_back(best_node);
    visited.emplace(best_node);
  }
  Array::reverse(&results);
  return results;
}

}  // namespace nmtkit
