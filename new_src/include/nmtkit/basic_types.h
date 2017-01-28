#ifndef NMTKIT_BASIC_TYPES_H_
#define NMTKIT_BASIC_TYPES_H_

#include <ostream>
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

  Token() {}
  explicit Token(const std::string surface_) : surface(surface_) {}
};

// Structure to represent one sentence.
struct Sentence {
  // List of tokens in the sentence by head-to-tail order.
  std::vector<Token> tokens;

  // Any-type features.
  FeatureMap features;

  Sentence() {}

  // Make a Sentence object from surface texts.
  //
  // Arguments:
  //   surfaces: List of surface texts.
  explicit Sentence(const std::vector<std::string> surfaces) {
    for (const std::string & surf : surfaces) {
      tokens.emplace_back(Token(surf));
    }
  }
};

inline bool operator==(const Token & a, const Token & b) {
  return a.surface == b.surface && a.features == b.features;
}

inline bool operator==(const Sentence & a, const Sentence & b) {
  return a.tokens == b.tokens && a.features == b.features;
}

std::ostream & operator<<(std::ostream & os, const FeatureMap & x);
std::ostream & operator<<(std::ostream & os, const Token & x);
std::ostream & operator<<(std::ostream & os, const Sentence & x);

}  // namespace nmtkit

#endif  // NMTKIT_BASIC_TYPES_H_
