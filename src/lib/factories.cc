#include "config.h"

#include <nmtkit/exception.h>
#include <nmtkit/factories.h>

#include <nmtkit/backward_encoder.h>
#include <nmtkit/bidirectional_encoder.h>
#include <nmtkit/forward_encoder.h>

#include <nmtkit/bahdanau_decoder.h>
#include <nmtkit/default_decoder.h>
#include <nmtkit/luong_decoder.h>

#include <nmtkit/bilinear_attention.h>
#include <nmtkit/mlp_attention.h>

#include <nmtkit/binary_code_predictor.h>
#include <nmtkit/softmax_predictor.h>

#include <nmtkit/frequency_code.h>

using namespace std;

namespace nmtkit {

boost::shared_ptr<Encoder> Factory::createEncoder(
    const boost::property_tree::ptree & config,
    const Vocabulary & vocab,
    dynet::Model * model) {
  const string name = config.get<string>("Model.encoder_type");
  const unsigned num_layers = config.get<unsigned>("Model.num_layers");
  const unsigned vocab_size = vocab.size();
  const unsigned embed_size = config.get<unsigned>(
      "Model.source_embedding_size");
  const unsigned hidden_size = config.get<unsigned>(
      "Model.encoder_hidden_size");

  if (name == "backward") {
    return boost::shared_ptr<Encoder>(
        new BackwardEncoder(
            num_layers, vocab_size, embed_size, hidden_size, model));
  } else if (name == "bidirectional") {
    return boost::shared_ptr<Encoder>(
        new BidirectionalEncoder(
            num_layers, vocab_size, embed_size, hidden_size, model));
  } else if (name == "forward") {
    return boost::shared_ptr<Encoder>(
        new ForwardEncoder(
            num_layers, vocab_size, embed_size, hidden_size, model));
  }
  NMTKIT_FATAL("Invalid encoder name: " + name);
}

boost::shared_ptr<Decoder> Factory::createDecoder(
    const boost::property_tree::ptree & config,
    const Vocabulary & vocab,
    const Encoder & encoder,
    dynet::Model * model) {
  const string name = config.get<string>("Model.decoder_type");
  const unsigned num_layers = config.get<unsigned>("Model.num_layers");
  const unsigned vocab_size = vocab.size();
  const unsigned in_embed_size = config.get<unsigned>(
      "Model.target_embedding_size");
  const unsigned out_embed_size = config.get<unsigned>(
      "Model.output_embedding_size");
  const unsigned hidden_size = config.get<unsigned>(
      "Model.decoder_hidden_size");
  const unsigned seed_size = encoder.getStateSize();
  const unsigned context_size = encoder.getOutputSize();

  if (name == "bahdanau") {
    return boost::shared_ptr<Decoder>(
        new BahdanauDecoder(
            num_layers, vocab_size, in_embed_size, out_embed_size, hidden_size,
            seed_size, context_size, model));
  } else if (name == "default") {
    return boost::shared_ptr<Decoder>(
        new DefaultDecoder(
            num_layers, vocab_size, in_embed_size, hidden_size,
            seed_size, context_size, model));
  } else if (name == "luong") {
    return boost::shared_ptr<Decoder>(
        new LuongDecoder(
            num_layers, vocab_size, in_embed_size, out_embed_size, hidden_size,
            seed_size, context_size, model));
  }
  NMTKIT_FATAL("Invalid decoder name: " + name);
}

boost::shared_ptr<Attention> Factory::createAttention(
    const boost::property_tree::ptree & config,
    const Encoder & encoder,
    dynet::Model * model) {
  const string name = config.get<string>("Model.attention_type");
  const unsigned context_size = encoder.getOutputSize();
  const unsigned controller_size = config.get<unsigned>(
      "Model.decoder_hidden_size");

  if (name == "bilinear") {
    return boost::shared_ptr<Attention>(
        new BilinearAttention(context_size, controller_size, model));
  } else if (name == "mlp") {
    const unsigned hidden_size = config.get<unsigned>(
        "Model.attention_hidden_size");
    return boost::shared_ptr<Attention>(
        new MLPAttention(context_size, controller_size, hidden_size, model));
  }
  NMTKIT_FATAL("Invalid attention name: " + name);
}

boost::shared_ptr<Predictor> Factory::createPredictor(
    const boost::property_tree::ptree & config,
    const Vocabulary & vocab,
    dynet::Model * /*model*/) {
  const string name = config.get<string>("Model.predictor_type");

  if (name == "binary_code") {
    boost::shared_ptr<BinaryCode> bc(new FrequencyCode(vocab));
    return boost::shared_ptr<Predictor>(new BinaryCodePredictor(bc));
  } else if (name == "softmax") {
    return boost::shared_ptr<Predictor>(new SoftmaxPredictor(vocab.size()));
  }
  NMTKIT_FATAL("Invalid predictor name: " + name);
}

}  // namespace nmtkit
