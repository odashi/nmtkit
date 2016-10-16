#include <nmtkit/batch_converter.h>

#include <algorithm>
#include <utility>

using namespace std;

namespace nmtkit {

BatchConverter::BatchConverter(
    const Vocabulary & src_vocab,
    const Vocabulary & trg_vocab)
  : src_bos_id_(src_vocab.getID("<s>")),
    src_eos_id_(src_vocab.getID("</s>")),
    trg_bos_id_(trg_vocab.getID("<s>")),
    trg_eos_id_(trg_vocab.getID("</s>"))
{}

void BatchConverter::convert(const vector<Sample> & samples, Batch * batch) {
  unsigned sl = 0, tl = 0, bs = samples.size();
  for (const Sample & s : samples) {
    sl = max(sl, static_cast<unsigned>(s.source.size()));
    tl = max(tl, static_cast<unsigned>(s.target.size()));
  }

  vector<vector<unsigned>> src(sl + 2, vector<unsigned>(bs));
  vector<vector<unsigned>> trg(tl + 2, vector<unsigned>(bs));

  for (unsigned i = 0; i < bs; ++i) {
    const Sample & s = samples[i];

    src[0][i] = src_bos_id_;
    src[sl + 1][i] = src_eos_id_;
    for (unsigned j = 0; j < sl; ++j) {
      src[j][i] = j < s.source.size() ? s.source[j] : 0;
    }

    trg[0][i] = trg_bos_id_;
    trg[tl + 1][i] = trg_eos_id_;
    for (unsigned j = 0; j < tl; ++j) {
      trg[j][i] = j < s.target.size() ? s.target[j] : 0;
    }
  }

  batch->source_id = std::move(src);
  batch->target_id = std::move(trg);
}

}  // namespace nmtkit

