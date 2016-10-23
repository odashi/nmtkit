#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <vector>
#include <nmtkit/inference_graph.h>

using namespace std;

namespace {

vector<nmtkit::InferenceGraph::Label> labels {
  {0, 0.0f},
  {1, 1.0f},
  {2, 2.0f},
};

}  // namespace

BOOST_AUTO_TEST_SUITE(InferenceGraphTest)

BOOST_AUTO_TEST_CASE(CheckAdding) {
  nmtkit::InferenceGraph graph;
  BOOST_CHECK_EQUAL(0, graph.size());

  graph.addNode(::labels[0]);
  BOOST_CHECK_EQUAL(1, graph.size());
  graph.addNode(::labels[1]);
  BOOST_CHECK_EQUAL(2, graph.size());
  graph.addNode(::labels[0]);
  BOOST_CHECK_EQUAL(3, graph.size());
  graph.addNode(::labels[2]);
  BOOST_CHECK_EQUAL(4, graph.size());
  graph.addNode(::labels[0]);
  BOOST_CHECK_EQUAL(5, graph.size());
  graph.addNode(::labels[1]);
  BOOST_CHECK_EQUAL(6, graph.size());
  graph.addNode(::labels[0]);
  BOOST_CHECK_EQUAL(7, graph.size());

  graph.clear();
  BOOST_CHECK_EQUAL(0, graph.size());

  graph.addNode(::labels[0]);
  BOOST_CHECK_EQUAL(1, graph.size());
}

BOOST_AUTO_TEST_CASE(CheckConnecting) {
  nmtkit::InferenceGraph graph;
  vector<nmtkit::InferenceGraph::Node *> nodes {
    graph.addNode(::labels[0]),
    graph.addNode(::labels[0]),
    graph.addNode(::labels[1]),
    graph.addNode(::labels[2]),
  };
  for (const auto* node : nodes) {
    BOOST_CHECK_EQUAL(0, node->prev().size());
    BOOST_CHECK_EQUAL(0, node->next().size());
  }

  graph.connect(nodes[0], nodes[1]);
  graph.connect(nodes[0], nodes[2]);
  graph.connect(nodes[1], nodes[2]);
  graph.connect(nodes[2], nodes[3]);
  vector<vector<const nmtkit::InferenceGraph::Node *>> expected_prev1 {
    {},
    {nodes[0]},
    {nodes[0], nodes[1]},
    {nodes[2]},
  };
  vector<vector<const nmtkit::InferenceGraph::Node *>> expected_next1 {
    {nodes[1], nodes[2]},
    {nodes[2]},
    {nodes[3]},
    {},
  };
  for (unsigned i = 0; i < nodes.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_prev1[i].begin(), expected_prev1[i].end(),
        nodes[i]->prev().begin(), nodes[i]->prev().end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_next1[i].begin(), expected_next1[i].end(),
        nodes[i]->next().begin(), nodes[i]->next().end());
  }

  graph.connect(nodes[3], nodes[0]);
  graph.connect(nodes[3], nodes[1]);
  graph.connect(nodes[3], nodes[2]);
  graph.connect(nodes[3], nodes[3]);  // self loop
  graph.connect(nodes[0], nodes[1]);  // duplicated connection
  vector<vector<const nmtkit::InferenceGraph::Node *>> expected_prev2 {
    {nodes[3]},
    {nodes[0], nodes[3]},
    {nodes[0], nodes[1], nodes[3]},
    {nodes[2], nodes[3]},
  };
  vector<vector<const nmtkit::InferenceGraph::Node *>> expected_next2 {
    {nodes[1], nodes[2]},
    {nodes[2]},
    {nodes[3]},
    {nodes[0], nodes[1], nodes[2], nodes[3]},
  };
  for (unsigned i = 0; i < nodes.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_prev2[i].begin(), expected_prev2[i].end(),
        nodes[i]->prev().begin(), nodes[i]->prev().end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected_next2[i].begin(), expected_next2[i].end(),
        nodes[i]->next().begin(), nodes[i]->next().end());
  }
}

BOOST_AUTO_TEST_CASE(CheckFinding) {
  nmtkit::InferenceGraph graph;
  vector<nmtkit::InferenceGraph::Node *> nodes {
    graph.addNode(::labels[0]),
    graph.addNode(::labels[0]),
    graph.addNode(::labels[1]),
    graph.addNode(::labels[2]),
  };
  graph.connect(nodes[0], nodes[2]);
  graph.connect(nodes[1], nodes[2]);
  graph.connect(nodes[2], nodes[3]);

  vector<const nmtkit::InferenceGraph::Node *> result;

  graph.findNodes(&result, [](const nmtkit::InferenceGraph::Node & node) {
      return node.label().word_id == 0;
  });
  BOOST_CHECK_EQUAL(2, result.size());
  BOOST_CHECK_EQUAL(nodes[0], result[0]);
  BOOST_CHECK_EQUAL(nodes[1], result[1]);
  
  graph.findNodes(&result, [](const nmtkit::InferenceGraph::Node & node) {
      return node.label().log_prob == 2.0;
  });
  BOOST_CHECK_EQUAL(1, result.size());
  BOOST_CHECK_EQUAL(nodes[3], result[0]);
  
  graph.findNodes(&result, [](const nmtkit::InferenceGraph::Node & node) {
      return node.prev().size() == 2 && node.next().size() == 1;
  });
  BOOST_CHECK_EQUAL(1, result.size());
  BOOST_CHECK_EQUAL(nodes[2], result[0]);

  graph.findNodes(&result, [](const nmtkit::InferenceGraph::Node & node) {
      return node.label().word_id == 100;
  });
  BOOST_CHECK_EQUAL(0, result.size());

  graph.clear();
  graph.findNodes(&result, [](const nmtkit::InferenceGraph::Node & node) {
      return node.label().word_id == 0;
  });
  BOOST_CHECK_EQUAL(0, result.size());
}

BOOST_AUTO_TEST_SUITE_END()

