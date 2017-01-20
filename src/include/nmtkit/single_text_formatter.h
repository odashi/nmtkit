#ifndef NMTKIT_SINGLE_TEXT_FORMATTER_H_
#define NMTKIT_SINGLE_TEXT_FORMATTER_H_

#include <nmtkit/formatter.h>
#include <fstream>

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

  void initialize(std::ostream * os) override {}
  void finalize(std::ostream * os) override {}
  void setReferencePath(std::string ref_file_path) override;

  void write(
      const std::string & source_line,
      const InferenceGraph & ig,
      const Vocabulary & source_vocab,
      const Vocabulary & target_vocab,
      std::ostream * os) override;

private:
  std::ifstream ref_ifs_;
};

}  // namespace nmtkit

#endif  // NMTKIT_SINGLE_TEXT_FORMATTER_H_
