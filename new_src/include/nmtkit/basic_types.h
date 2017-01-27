#ifndef NMTKIT_BASIC_TYPES_H_
#define NMTKIT_BASIC_TYPES_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace nmtkit {

// Structure to represent feature list.
typedef std::unordered_map<std::string, std::string> FeatureMap;

// Structure to represent one token.
struct Token {
  // Surface text of the token.
  std::string surface;

  // Any-type features.
  FeatureMap features;

  bool operator==(const Token & lhs) {
    return surface == src.surface && features == src.features;
  }
};

// Structure to represent one sentence.
struct Sentence {
  // List of tokens in the sentence by head-to-tail order.
  std::vector<Token> tokens;

  // Any-type features.
  FeatureMap features;

  bool operator==(const Sentence & lhs) {
    return tokens == src.tokens && features == src.features;
  }
};

}  // namespace nmtkit

#endif  // NMTKIT_BASIC_TYPES_H_
