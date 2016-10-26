#ifndef NMTKIT_SINGLE_TEXT_FORMATTER_H_
#define NMTKIT_SINGLE_TEXT_FORMATTER_H_

#include <nmtkit/formatter.h>

namespace nmtkit {

// Formatter class to output only tokens in each line.
class SingleTextFormatter : public Formatter {
  SingleTextFormatter(const SingleTextFormatter &) = delete;
  SingleTextFormatter(SingleTextFormatter &&) = delete;
  SingleTextFormatter & operator=(const SingleTextFormatter &) = delete;
  SingleTextFormatter & operator=(SingleTextFormatter &&) = delete;

public:
  SingleTextFormatter() {}
  ~SingleTextFormatter() override {}

  void write(
      const InferenceGraph & ig,
      const Vocabulary & vocab,
      std::ostream * os) override;
};

}  // namespace nmtkit

#endif  // NMTKIT_SINGLE_TEXT_FORMATTER_H_
