#ifndef NMTKIT_INFERENCE_GRAPH_H_
#define NMTKIT_INFERENCE_GRAPH_H_

#include <functional>
#include <vector>

namespace nmtkit {

class InferenceGraph {
  InferenceGraph(const InferenceGraph &) = delete;
  InferenceGraph(InferenceGraph &&) = delete;
  InferenceGraph & operator=(const InferenceGraph &) = delete;
  InferenceGraph & operator=(InferenceGraph &&) = delete;

public:
  struct Label {
    unsigned word_id;
    float word_log_prob;
    float accum_log_prob;
    std::vector<float> atten_probs;
  };

  class Node {
    friend class InferenceGraph;

    Node() = delete;
    Node(const Node &) = delete;
    Node(Node &&) = delete;
    Node & operator=(const Node &) = delete;
    Node & operator=(Node &&) = delete;

  public:
    Node(const Label & label) : label_(label) {}
    ~Node() {}

    const Label & label() const { return label_; }
    const std::vector<const Node *> & prev() const { return prev_; }
    const std::vector<const Node *> & next() const { return next_; }

  private:
    Label label_;
    std::vector<const Node *> prev_;
    std::vector<const Node *> next_;
  };

  InferenceGraph() {}
  ~InferenceGraph();

  // Clear the graph.
  void clear();

  // Adds new node into the graph.
  //
  // Arguments:
  //   label: Inner value of the node.
  //
  // Returns:
  //   Pointer of the new node. Users must not delete this pointer themselves.
  Node * addNode(const Label & label);

  // Connects two nodes in directional order.
  //
  // Arguments:
  //   prev: Pointer of the node in the previous side.
  //   next: Pointer of the node in the next side.
  void connect(Node * prev, Node * next);

  // Finds nodes which satisfy the condition.
  //
  // Arguments:
  //   result: Placeholder of the result nodes. Old data would be deleted before
  //           storing new data.
  //   cond: Predicate of the condition.
  //
  // Returns:
  //   List of found nodes.
  std::vector<const Node *> findNodes(
      std::function<bool(const Node &)> cond) const;

  // Finds the one-best full setence path.
  //
  // Arguments:
  //   bos_id: Word ID of "<s>".
  //   eos_id: Word ID of "</s>".
  //
  // Returns:
  //   List of nodes from "<s>" to "</s>" in the one-best path.
  std::vector<const Node *> findOneBestPath(
      const unsigned bos_id,
      const unsigned eos_id) const;

  // Retrieves number of inner nodes.
  //
  // Returns:
  //   Number of inner nodes.
  unsigned size() const { return nodes_.size(); }

private:
  std::vector<Node *> nodes_;
};

}  // namespace nmtkit

#endif  // NMTKIT_INFERENCE_GRAPH_H_
