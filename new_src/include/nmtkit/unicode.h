#ifndef NMTKIT_UNICODE_H_
#define NMTKIT_UNICODE_H_

#include <string>
#include <vector>

namespace nmtkit {

// UTF-8 utilitiy functions.
class UTF8 {
  UTF8() = delete;
  UTF8(const UTF8 &) = delete;
  UTF8(UTF8 &&) = delete;
  UTF8 & operator=(const UTF8 &) = delete;
  UTF8 & operator=(UTF8 &&) = delete;

public:
  // Check whether the character is a UTF-8 first byte or not.
  //
  // Arguments:
  //   c: Target character.
  //
  // Returns:
  //   true if `c` is a first byte, false otherwise.
  static bool isFirstByte(const unsigned char c);

  // Retrieves expected number of bytes in the UTF-8 letter begun by given
  // first byte.
  //
  // Arguments:
  //   first_byte: The UTF-8 first byte.
  //
  // Returns:
  //   Expected number of bytes in the UTF-8 letter.
  static unsigned getNumBytes(const unsigned char first_byte);

  // Separates UTF-8 string into letters.
  //
  // Arguments:
  //   str: Target string.
  //
  // Returns:
  //   A list of UTF-8 letters.
  static std::vector<std::string> getLetters(const std::string & str);
};

}  // namespace nmtkit

#endif  // NMTKIT_UNICODE_H_
