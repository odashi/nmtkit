#include "config.h"

#include <nmtkit/single_text_formatter.h>

#include <nmtkit/exception.h>

using namespace std;

namespace nmtkit {

void SingleTextFormatter::write(
    const vector<string> & source_words,
    const InferenceGraph & ig,
    const Vocabulary & source_vocab,
    const Vocabulary & target_vocab,
    std::ostream * os) {
  const unsigned bos_id = target_vocab.getID("<s>");
  const unsigned eos_id = target_vocab.getID("</s>");

  // Note: this formatter outputs only the one-best result.
  
  // Finds <s> node.
  vector<const nmtkit::InferenceGraph::Node *> heads;
  ig.findNodes(&heads, [&](const nmtkit::InferenceGraph::Node & node) {
      return node.label().word_id == bos_id;
  });
  NMTKIT_CHECK_EQ(
      1, heads.size(), "No or multiple '<s>' nodes in the inference graph.");
  const nmtkit::InferenceGraph::Node * cur_node = heads[0];

  for (unsigned num_words = 0; ; ++num_words) {
    // Finds the most accurate word.
    NMTKIT_CHECK(!cur_node->next().empty(), "No next node of the current node");
    const nmtkit::InferenceGraph::Node * best_node = nullptr;
    float best_log_prob = -1e100;
    for (const auto next_node : cur_node->next()) {
      if (next_node->label().word_log_prob > best_log_prob) {
        best_node = next_node;
        best_log_prob = next_node->label().word_log_prob;
      }
    }
    cur_node = best_node;

    // Outputs the word or exit.
    const unsigned word_id = cur_node->label().word_id;
    if (word_id == eos_id) {
      break;
    }
    if (num_words > 0) {
      *os << ' ';
    }
    *os << target_vocab.getWord(word_id);
  }
  *os << endl;
}

}  // namespace nmtkit
