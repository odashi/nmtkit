#ifndef NMTKIT_FORMATTER_H_
#define NMTKIT_FORMATTER_H_

#include <ostream>
#include <string>
#include <vector>
#include <nmtkit/inference_graph.h>
#include <nmtkit/vocabulary.h>

namespace nmtkit {

// Abstract class for output formatters.
//
// Basic usage:
//   ADerivedFormatter formatter(...);
//   formatter.initialize(&os);
//   for (const auto & ig : inference_graphs) {
//     formatter.write(ig, vocab, &os);
//   }
//   formatter.finalize(&os);
class Formatter {
  Formatter(const Formatter &) = delete;
  Formatter(Formatter &&) = delete;
  Formatter & operator=(const Formatter &) = delete;
  Formatter & operator=(Formatter &&) = delete;

public:
  Formatter() {}
  virtual ~Formatter() {}

  // Outputs the initial information.
  //
  // Arguments:
  //   os: Target output stream.
  virtual void initialize(std::ostream * os) = 0;
  
  // Outputs the final information.
  //
  // Arguments:
  //   os: Target output stream.
  virtual void finalize(std::ostream * os) = 0;

  // Writes output information into a stream.
  //
  // Arguments:
  //   source_words: List of source words.
  //   ig: Inference graph object which has the output information.
  //   source_vocab: Vocabulary object for the source language.
  //   target_vocab: Vocabulary object for the target language.
  //   os: Target output stream.
  virtual void write(
      const std::vector<std::string> & source_words,
      const InferenceGraph & ig,
      const Vocabulary & source_vocab,
      const Vocabulary & target_vocab,
      std::ostream * os) = 0;
};

}  // namespace nmtkit

#endif  // NMTKIT_FORMATTER_H_
