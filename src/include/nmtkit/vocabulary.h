#ifndef NMTKIT_VOCABULARY_H_
#define NMTKIT_VOCABULARY_H_

#include <map>
#include <string>
#include <vector>

namespace NMTKit {

class Vocabulary {
  public:
    explicit Vocabulary(const std::string &vocab_filename);
    Vocabulary(const std::string &corpus_filename, int size);

    void save(const std::string &vocab_filename);

    int getID(const std::string &word) const;
    std::string getWord(int id) const;

    int size() const;
  
  private:
    std::map<std::string, int> stoi_;
    std::vector<std::string> itos_;
};

}  // namespace NMTKit

#endif  // NMTKIT_VOCABULARY_H_

