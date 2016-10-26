#ifndef NMTKIT_FORMATTER_H_
#define NMTKIT_FORMATTER_H_

#include <ostream>
#include <nmtkit/inference_graph.h>
#include <nmtkit/vocabulary.h>

namespace nmtkit {

// Abstract class for output formatters.
class Formatter {
  Formatter(const Formatter &) = delete;
  Formatter(Formatter &&) = delete;
  Formatter & operator=(const Formatter &) = delete;
  Formatter & operator=(Formatter &&) = delete;

public:
  Formatter() {}
  virtual ~Formatter() {}

  // Writes output information into a stream.
  //
  // Arguments:
  //   ig: Inference graph object which has the output information.
  //   vocab: Vocabulary object for the target language.
  //   os: Target output stream.
  virtual void write(
      const InferenceGraph & ig,
      const Vocabulary & vocab,
      std::ostream * os) = 0;
};

}  // namespace nmtkit

#endif  // NMTKIT_FORMATTER_H_
