#ifndef NMTKIT_READER_H_
#define NMTKIT_READER_H_

#include <nmtkit/basic_types.h>

namespace nmtkit {

// Interface of the data reader.
class Reader {
  Reader(const Reader &) = delete;
  Reader(Reader &&) = delete;
  Reader & operator=(const Reader &) = delete;
  Reader & operator=(Reader &&) = delete;

 public:
  Reader() {}
  virtual ~Reader() {}

  // Read one sample from the stream.
  //
  // Arguments:
  //   sp: Placeholder to store next sample.
  //
  // Returns:
  //   true if reading next sample is succeeded, false otherwise.
  virtual bool read(SentencePair * sp) = 0;
};

}  // namespace nmtkit

#endif  // NMTKIT_READER_H_
