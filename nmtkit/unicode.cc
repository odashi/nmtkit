#include <nmtkit/exception.h>
#include <nmtkit/unicode.h>

using std::string;
using std::vector;

namespace nmtkit {

bool UTF8::isFirstByte(const unsigned char c) {
  return !(c & 0x80) || (c >= 0xc2 && c <= 0xfd);
}

unsigned UTF8::getNumBytes(const unsigned char first_byte) {
  if ((first_byte & 0x80) == 0x00) return 1;
  if ((first_byte & 0xe0) == 0xc0) return 2;
  if ((first_byte & 0xf0) == 0xe0) return 3;
  if ((first_byte & 0xf8) == 0xf0) return 4;
  if ((first_byte & 0xfc) == 0xf8) return 5;
  if ((first_byte & 0xfe) == 0xfc) return 6;
  NMTKIT_FATAL("Invalid UTF-8 first byte.");
}

vector<string> UTF8::getLetters(const string & str) {
  const unsigned len = str.size();
  unsigned prev = 0;
  vector<string> letters;
  while (prev < len) {
    NMTKIT_CHECK(isFirstByte(str[prev]), "Invalid UTF-8 sequence.");
    const unsigned n = getNumBytes(str[prev]);
    NMTKIT_CHECK(prev + n <= len, "Invalid UTF-8 sequence.");
    for (unsigned i = 1; i < n; ++i) {
      NMTKIT_CHECK(!isFirstByte(str[prev + i]), "Invalid UTF-8 sequence.");
    }
    letters.emplace_back(str.substr(prev, n));
    prev += n;
  }
  return letters;
}

}  // namespace nmtkit
