#ifndef NMTKIT_CORPUS_H_
#define NMTKIT_CORPUS_H_

#include <string>
#include <vector>
#include <nmtkit/vocabulary.h>

namespace NMTKit {

class Corpus {
  Corpus() = delete;
  Corpus(const Corpus &) = delete;
  Corpus(Corpus &&) = delete;
  Corpus & operator=(const Corpus &) = delete;
  Corpus & operator=(Corpus &&) = delete;

public:
  // Loads all samples in the tokenized corpus.
  // Arguments:
  //   filepath: Location of the corpus file.
  //   vocab: Vocabulary object for the corpus language.
  //   result: Placeholder to store new samples. Old data will be deleted
  //           automatically before storing new samples.
  static void loadFromTokenFile(
      const std::string & filepath,
      const Vocabulary & vocab,
      std::vector<std::vector<unsigned>> * result);
};

}  // namespace NMTKit

#endif  // NMTKIT_CORPUS_H_

