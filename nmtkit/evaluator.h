#ifndef NMTKIT_EVALUATOR_H_
#define NMTKIT_EVALUATOR_H_

#include <boost/serialization/access.hpp>
#include <nmtkit/encoder_decoder.h>

namespace nmtkit {

// Abstract class to implement translation quality evaluator interface.
class Evaluator {
  Evaluator(const Evaluator &) = delete;
  Evaluator(Evaluator &&) = delete;
  Evaluator & operator=(const Evaluator &) = delete;
  Evaluator & operator=(Evaluator &&) = delete;

public:
  Evaluator() {}
  virtual ~Evaluator() {}

  // Evaluate translation quality of an encoder-decoder object.
  //
  // Arguments:
  //   encdec: Target EncoderDecoder object.
  //
  // Returns:
  //   An evaluation score.
  virtual float evaluate(EncoderDecoder * encdec) = 0;
};

} // namespace nmtkit

#endif // NMTKIT_EVALUATOR_H_
