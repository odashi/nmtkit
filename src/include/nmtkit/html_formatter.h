#ifndef NMTKIT_HTML_FORMATTER_H_
#define NMTKIT_HTML_FORMATTER_H_

#include <nmtkit/formatter.h>

namespace nmtkit {

// Formatter class to output html document with detailed information of the
// inference graph.
class HTMLFormatter : public Formatter {
  HTMLFormatter(const HTMLFormatter &) = delete;
  HTMLFormatter(HTMLFormatter &&) = delete;
  HTMLFormatter & operator=(const HTMLFormatter &) = delete;
  HTMLFormatter & operator=(HTMLFormatter &&) = delete;

public:
  HTMLFormatter();
  ~HTMLFormatter() override {}

  void initialize(std::ostream * os) override;
  void finalize(std::ostream * os) override;

  void write(
      const std::string & source_line,
      const InferenceGraph & ig,
      const Vocabulary & source_vocab,
      const Vocabulary & target_vocab,
      std::ostream * os) override;

private:
  unsigned num_output_;
};

}  // namespace nmtkit

#endif  // NMTKIT_HTML_FORMATTER_H_
