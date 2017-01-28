#ifndef NMTKIT_SIMPLE_TEXT_READER_H_
#define NMTKIT_SIMPLE_TEXT_READER_H_

#include <nmtkit/reader.h>
#include <fstream>
#include <string>

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
  //   sfilepath: Location to the token file.
  explicit SimpleTextReader(const std::string & filepath);

  ~SimpleTextReader() override;

  bool read(Sentence * sentence) override;

 private:
  std::ifstream ifs_;
};

}  // namespace nmtkit

#endif  // NMTKIT_SIMPLE_TEXT_READER_H_
