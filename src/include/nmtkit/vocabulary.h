#ifndef NMTKIT_VOCABULARY_H_
#define NMTKIT_VOCABULARY_H_

#include <map>
#include <string>
#include <vector>

namespace nmtkit {

class Vocabulary {
  Vocabulary() = delete;
  Vocabulary(const Vocabulary &) = delete;
  Vocabulary(Vocabulary &&) = delete;
  Vocabulary & operator=(const Vocabulary &) = delete;
  Vocabulary & operator=(Vocabulary &&) = delete;

public:
  explicit Vocabulary(const std::string & vocab_filename);
  Vocabulary(const std::string & corpus_filename, unsigned size);

  void save(const std::string & vocab_filename) const;

  unsigned getID(const std::string & word) const;
  std::string getWord(unsigned id) const;

  unsigned size() const;

private:
  std::map<std::string, unsigned> stoi_;
  std::vector<std::string> itos_;
};

}  // namespace nmtkit

#endif  // NMTKIT_VOCABULARY_H_
