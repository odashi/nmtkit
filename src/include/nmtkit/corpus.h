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
  static void loadSingleSentences(
      const std::string & filepath,
      const Vocabulary & vocab,
      std::vector<std::vector<unsigned>> * result);

  // Loads tokenized parallel corpus.
  // Arguments:
  //   src_filepath: Location of the source corpus file.
  //   trg_filepath: Location of the target corpus file.
  //   src_vocab: Vocabulary object for the source language.
  //   trg_vocab: Vocabulary object for the target language.
  //   max_length: Maximum number of words in a sentence. Samples which exceeds
  //               this value will be skipped.
  //   src_result: Placeholder to store new source samples. Old data will be
  //               deleted automatically before storing new samples.
  //   trg_result: Placeholder to store new target samples. Old data will be
  //               deleted automatically before storing new samples.
  static void loadParallelSentences(
      const std::string & src_filepath,
      const std::string & trg_filepath,
      const Vocabulary & src_vocab,
      const Vocabulary & trg_vocab,
      unsigned max_length,
      std::vector<std::vector<unsigned>> * src_result,
      std::vector<std::vector<unsigned>> * trg_result);
};

}  // namespace NMTKit

#endif  // NMTKIT_CORPUS_H_

