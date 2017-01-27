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

  Token()
    : surface(), features() {}
  Token(const Token & src)
    : surface(src.surface), features(src.features) {}
  Token(Token && src)
    : surface(std::move(src.surface)), features(std::move(src.features)) {}
  explicit Token(const std::string & surface_)
    : surface(surface_), features() {}
  Token(const std::string & surface_, const FeatureMap & features_)
    : surface(surface_), features(features_) {}

  Token & operator=(const Token & src) {
    surface = src.surface;
    features = src.features;
  }
  Token & operator=(Token && src) {
    surface = std::move(src.surface);
    features = std::move(src.features);
  }

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

  Sentence()
    : tokens(), features() {}
  Sentence(const Sentence & src)
    : tokens(src.tokens), features(src.features) {}
  Sentence(Sentence && src)
    : tokens(std::move(src.tokens)), features(std::move(src.features)) {}
  explicit Sentence(const std::vector<Token> & tokens_)
    : tokens(tokens_), features() {}
  Sentence(const std::vector<Token> & tokens_, const FeatureMap & features)
    : tokens(tokens_), features(features_) {}

  Sentence & operator=(const Sentence & src) {
    tokens = src.tokens;
    features = src.features;
  }
  Sentence & operator=(Sentence && src) {
    tokens = std::move(src.tokens);
    features = std::move(src.features);
  }

  bool operator==(const Sentence & lhs) {
    return tokens == src.tokens && features == src.features;
  }
};

}  // namespace nmtkit

#endif  // NMTKIT_BASIC_TYPES_H_
