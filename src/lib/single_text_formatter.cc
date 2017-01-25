#include "config.h"

#include <nmtkit/single_text_formatter.h>

#include <nmtkit/exception.h>

using std::string;
using std::vector;

namespace nmtkit {

void SingleTextFormatter::write(
    const string & source_line,
    const string & ref_line,
    const InferenceGraph & ig,
    const Vocabulary & source_vocab,
    const Vocabulary & target_vocab,
    std::ostream * os) {
  const unsigned bos_id = target_vocab.getID("<s>");
  const unsigned eos_id = target_vocab.getID("</s>");

  // Note: this formatter outputs only the one-best result.
  auto nodes = ig.findOneBestPath(bos_id, eos_id);

  // Note: Ignores <s> and </s>.
  vector<unsigned> word_ids;
  for (unsigned i = 1; i < nodes.size() - 1; ++i) {
    word_ids.emplace_back(nodes[i]->label().word_id);
  }

  *os << target_vocab.convertToSentence(word_ids) << std::endl;
}

}  // namespace nmtkit
