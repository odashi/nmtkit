#include <nmtkit/batch_converter.h>

#include <algorithm>

using namespace std;

namespace nmtkit {

void BatchConverter::convert(
    const vector<Sample> & samples,
    unsigned eos_id,
    Batch * batch) {
  unsigned sl = 0, tl = 0, bs = samples.size();
  for (const Sample & s : samples) {
    sl = max(sl, static_cast<unsigned>(s.source.size()));
    tl = max(tl, static_cast<unsigned>(s.target.size()));
  }
  batch->source_id = vector<vector<unsigned>>(sl, vector<unsigned>(bs));
  batch->target_id = vector<vector<unsigned>>(tl, vector<unsigned>(bs));
  for (unsigned i = 0; i < bs; ++i) {
    const Sample & s = samples[i];
    for (unsigned j = 0; j < sl; ++j) {
      batch->source_id[j][i] = j < s.source.size() ? s.source[j] : 0;
    }
    for (unsigned j = 0; j < tl; ++j) {
      batch->target_id[j][i] = j < s.target.size() ? s.target[j] : 0;
    }
  }
}

}  // namespace nmtkit

