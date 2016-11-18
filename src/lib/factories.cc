#include "config.h"

#include <nmtkit/exception.h>
#include <nmtkit/factories.h>

#include <nmtkit/backward_encoder.h>
#include <nmtkit/bidirectional_encoder.h>
#include <nmtkit/forward_encoder.h>

#include <nmtkit/default_decoder.h>

#include <nmtkit/bilinear_attention.h>
#include <nmtkit/mlp_attention.h>

using namespace std;

namespace nmtkit {

boost::shared_ptr<Encoder> Factory::createEncoder(
    const string & name,
    const unsigned vocab_size,
    const unsigned embed_size,
    const unsigned hidden_size,
    dynet::Model * model) {
  if (name == "backward") {
    return boost::shared_ptr<Encoder>(
        new BackwardEncoder(
            1, vocab_size, embed_size, hidden_size, model));
  } else if (name == "bidirectional") {
    return boost::shared_ptr<Encoder>(
        new BidirectionalEncoder(
            1, vocab_size, embed_size, hidden_size, model));
  } else if (name == "forward") {
    return boost::shared_ptr<Encoder>(
        new ForwardEncoder(
            1, vocab_size, embed_size, hidden_size, model));
  }
  NMTKIT_FATAL("Invalid encoder name: " + name);
}

boost::shared_ptr<Decoder> Factory::createDecoder(
    const string & name,
    const unsigned vocab_size,
    const unsigned in_embed_size,
    const unsigned out_embed_size,
    const unsigned hidden_size,
    const unsigned seed_size,
    const unsigned context_size,
    dynet::Model * model) {
  if (name == "default") {
    return boost::shared_ptr<Decoder>(
        new DefaultDecoder(
            vocab_size, in_embed_size, hidden_size,
            seed_size, context_size, model));
  }
  NMTKIT_FATAL("Invalid decoder name: " + name);
}

boost::shared_ptr<Attention> Factory::createAttention(
    const string & name,
    const unsigned context_size,
    const unsigned controller_size,
    const unsigned hidden_size,
    dynet::Model * model) {
  if (name == "bilinear") {
    return boost::shared_ptr<Attention>(
        new BilinearAttention(
            context_size, controller_size, model));
  } else if (name == "mlp") {
    return boost::shared_ptr<Attention>(
        new MLPAttention(
            context_size, controller_size, hidden_size, model));
  }
  NMTKIT_FATAL("Invalid attention name: " + name);
}

}  // namespace nmtkit
