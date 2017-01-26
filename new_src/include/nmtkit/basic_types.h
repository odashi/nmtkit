#ifndef NMTKIT_BASIC_TYPES_H_
#define NMTKIT_BASIC_TYPES_H_

#include <string>
#include <vector>
#include <unordered_map>

namespace nmtkit {

// Structure to represent feature list.
typedef std::unordered_map<std::string, std::string> FeatureMap;

// Structure to represent one token.
struct Token {
  std::string surface;
  FeatureMap features;
};

// Structure to represent one sentence.
struct Sentence {
  std::vector<Token> tokens;
  FeatureMap features;
};

// Structure to represent one sentence pair.
struct SentencePair {
  Sentence source;
  Sentence target;
  FeatureMap features;
};

}  // namespace nmtkit

#endif  // NMTKIT_BASIC_TYPES_H_
