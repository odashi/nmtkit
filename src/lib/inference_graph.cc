#include <nmtkit/inference_graph.h>

#include <algorithm>
#include <nmtkit/exception.h>

using namespace std;

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

void InferenceGraph::findNodes(
    vector<const Node *> * result,
    function<bool(const Node &)> cond) const {
  result->clear();
  for (const Node * node : nodes_) {
    if (cond(*node)) {
      result->emplace_back(node);
    }
  }
}

}  // namespace nmtkit
