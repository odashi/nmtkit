#include <nmtkit/basic_types.h>

#include <config.h>

namespace nmtkit {

std::ostream & operator<<(std::ostream & os, const FeatureMap & x) {
  os << "FeatureMap {";
  bool first_time = true;
  for (const auto & p : x) {
    if (!first_time) os << ", ";
    first_time = false;
    os << "\"" << p.first << "\": \"" << p.second << "\"";
  }
  os << "}";
  return os;
}

std::ostream & operator<<(std::ostream & os, const Token & x) {
  os << "Token {surface: \"" << x.surface
     << "\", features: " << x.features << "}";
  return os;
}

std::ostream & operator<<(std::ostream & os, const Sentence & x) {
  os << "Sentence {tokens: [";
  bool first_time = true;
  for (const Token & tok : x.tokens) {
    if (!first_time) os << ", ";
    first_time = false;
    os << tok;
  }
  os << "], features: " << x.features << "}";
  return os;
}

}  // namespace nmtkit
