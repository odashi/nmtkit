#ifndef NMTKIT_BASIC_TYPES_H_
#define NMTKIT_BASIC_TYPES_H_

#include <string>
#include <vector>
#include <map>

namespace nmtkit {

// Structure to represent one token.
struct Token {
  std::string surface;
  std::unordered_map<std::string, std::string> features;
};

// Structure to represent one sentence.
struct Sentence {
  std::vector<Token> tokens;
  std::unordered_map<std:string, std::string> features;
};

// Structure to represent one sentence pair.
struct SentencePair {
  Sentence source;
  Sentence target;
  std::unordered_map<std::string, std::string> features;
};

}  // namespace nmtkit

#endif  // NMTKIT_BASIC_TYPES_H_
