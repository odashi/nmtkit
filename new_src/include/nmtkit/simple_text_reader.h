#ifndef NMTKIT_SIMPLE_TEXT_READER_H_
#define NMTKIT_SIMPLE_TEXT_READER_H_

#include <fstream>
#include <string>
#include <nmtkit/reader.h>

namespace nmtkit {

// Reader for whitespace-separated tokens.
class SimpleTextReader : public Reader {
  SimpleTextReader(const SimpleTextReader &) = delete;
  SimpleTextReader(SimpleTextReader &&) = delete;
  SimpleTextReader & operator=(const SimpleTextReader &) = delete;
  SimpleTextReader & operator=(SimpleTextReader &&) = delete;

public:
  // Creates new reader object.
  //
  // Arguments:
  //   src_filepath: Location to the source token file.
  //                 If this argument is empty, source sentences would not be
  //                 included in returned samples.
  //                 at least this argument or `trg_filepath` should be set as
  //                 non-empty.
  //   trg_filepath: Location to the target token file.
  //                 If this argument is empty, target sentences would not be
  //                 included in returned samples.
  //                 at least this argument or `src_filepath` should be set as
  //                 non-empty.
  SimpleTextReader(
      const std::string & src_filepath,
      const std::string & trg_filepath);

  ~SimpleTextReader() override;

  bool read(SentencePair * sp) override;

private:
  std::ifstream src_ifs_;
  std::ifstream trg_ifs_;
};

}  // namespace nmtkit

#endif  // NMTKIT_SIMPLE_TEXT_READER_H_
