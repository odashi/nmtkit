#include <config.h>

#include <nmtkit/exception.h>
#include <nmtkit/factories.h>

#include <nmtkit/character_vocabulary.h>
#include <nmtkit/word_vocabulary.h>
#include <nmtkit/bpe_vocabulary.h>

#include <nmtkit/backward_encoder.h>
#include <nmtkit/bidirectional_encoder.h>
#include <nmtkit/forward_encoder.h>

#include <nmtkit/bahdanau_decoder.h>
#include <nmtkit/default_decoder.h>
#include <nmtkit/luong_decoder.h>

#include <nmtkit/bilinear_attention.h>
#include <nmtkit/mlp_attention.h>

#include <nmtkit/binary_code_predictor.h>
#include <nmtkit/hybrid_predictor.h>
#include <nmtkit/softmax_predictor.h>

#include <nmtkit/huffman_code.h>
#include <nmtkit/frequency_code.h>

#include <nmtkit/convolutional_ecc.h>
#include <nmtkit/identity_ecc.h>

using std::make_shared;
using std::shared_ptr;
using std::string;

namespace {

// Creates new BinaryCode object.
//
// Arguments:
//   config: ptree object with correctly-defined options.
//   vocab: Vocabulary object for the target language.
//
// Returns:
//   A shared pointer of BinaryCode object.
shared_ptr<nmtkit::BinaryCode> createBinaryCode(
    const boost::property_tree::ptree & config,
    const nmtkit::Vocabulary & vocab) {
  const string name = config.get<string>("Model.binary_code_type");

  if (name == "huffman") {
    return make_shared<nmtkit::HuffmanCode>(vocab);
  } else if (name == "frequency") {
    return make_shared<nmtkit::FrequencyCode>(vocab);
  }
  NMTKIT_FATAL("Invalid name of binary code: " + name);
}

// Creates new ErrorCorrectingCode object.
//
// Arguments:
//   config: ptree object with correctly-defined options.
//
// Returns:
//   A shared pointer of ErrorCorrectingCode object.
shared_ptr<nmtkit::ErrorCorrectingCode> createErrorCorrectingCode(
    const boost::property_tree::ptree & config) {
  const string name = config.get<string>("Model.error_correcting_code_type");

  if (name == "convolutional") {
    const unsigned num_registers = config.get<unsigned>(
        "Model.convolutional_ecc_num_registers");
    return make_shared<nmtkit::ConvolutionalECC>(num_registers);
  } else if (name == "identity") {
    return make_shared<nmtkit::IdentityECC>();
  }
  NMTKIT_FATAL("Invalid name of error correcting code: " + name);
}

}  // namespace

namespace nmtkit {

shared_ptr<Vocabulary> Factory::createVocabulary(
    const string & corpus_filepath,
    const string & type,
    const unsigned size) {
  if (type == "bpe") {
    return make_shared<BPEVocabulary>(corpus_filepath, size);
  } else if (type == "character") {
    return make_shared<CharacterVocabulary>(corpus_filepath, size);
  } else if (type == "word") {
    return make_shared<WordVocabulary>(corpus_filepath, size);
  }
  NMTKIT_FATAL("Invalid vocabulary type: " + type);
}

shared_ptr<Encoder> Factory::createEncoder(
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
  const float dropout_rate = config.get<float>("Train.dropout_rate");

  if (name == "backward") {
    return make_shared<BackwardEncoder>(
        num_layers, vocab_size, embed_size, hidden_size, dropout_rate, model);
  } else if (name == "bidirectional") {
    return make_shared<BidirectionalEncoder>(
        num_layers, vocab_size, embed_size, hidden_size, dropout_rate, model);
  } else if (name == "forward") {
    return make_shared<ForwardEncoder>(
        num_layers, vocab_size, embed_size, hidden_size, dropout_rate, model);
  }
  NMTKIT_FATAL("Invalid encoder name: " + name);
}

shared_ptr<Decoder> Factory::createDecoder(
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
  const float dropout_rate = config.get<float>("Train.dropout_rate");

  if (name == "bahdanau") {
    return make_shared<BahdanauDecoder>(
        num_layers, vocab_size, in_embed_size, out_embed_size, hidden_size,
        seed_size, context_size, dropout_rate, model);
  } else if (name == "default") {
    return make_shared<DefaultDecoder>(
        num_layers, vocab_size, in_embed_size, hidden_size,
        seed_size, context_size, dropout_rate, model);
  } else if (name == "luong") {
    return make_shared<LuongDecoder>(
        num_layers, vocab_size, in_embed_size, out_embed_size, hidden_size,
        seed_size, context_size, dropout_rate, model);
  }
  NMTKIT_FATAL("Invalid decoder name: " + name);
}

shared_ptr<Attention> Factory::createAttention(
    const boost::property_tree::ptree & config,
    const Encoder & encoder,
    dynet::Model * model) {
  const string name = config.get<string>("Model.attention_type");
  const unsigned context_size = encoder.getOutputSize();
  const unsigned controller_size = config.get<unsigned>(
      "Model.decoder_hidden_size");

  if (name == "bilinear") {
    return make_shared<BilinearAttention>(context_size, controller_size, model);
  } else if (name == "mlp") {
    const unsigned hidden_size = config.get<unsigned>(
        "Model.attention_hidden_size");
    return make_shared<MLPAttention>(
        context_size, controller_size, hidden_size, model);
  }
  NMTKIT_FATAL("Invalid attention name: " + name);
}

shared_ptr<Predictor> Factory::createPredictor(
    const boost::property_tree::ptree & config,
    const Vocabulary & vocab,
    const Decoder & decoder,
    dynet::Model * model) {
  const string name = config.get<string>("Model.predictor_type");

  if (name == "binary") {
    auto bc = ::createBinaryCode(config, vocab);
    auto ecc = ::createErrorCorrectingCode(config);
    const string loss_type = config.get<string>("Model.binary_code_loss_type");
    return make_shared<BinaryCodePredictor>(
        decoder.getOutputSize(), bc, ecc, loss_type, model);
  } else if (name == "hybrid") {
    auto bc = ::createBinaryCode(config, vocab);
    auto ecc = ::createErrorCorrectingCode(config);
    const unsigned softmax_size = config.get<unsigned>(
        "Model.hybrid_softmax_size");
    const string loss_type = config.get<string>("Model.binary_code_loss_type");
    const float softmax_weight = config.get<float>(
        "Model.hybrid_softmax_weight");
    const float binary_weight = config.get<float>(
        "Model.hybrid_binary_weight");
    return make_shared<HybridPredictor>(
        decoder.getOutputSize(), softmax_size, bc, ecc,
        loss_type, softmax_weight, binary_weight, model);
  } else if (name == "softmax") {
    return make_shared<SoftmaxPredictor>(
        decoder.getOutputSize(), vocab.size(), model);
  }
  NMTKIT_FATAL("Invalid predictor name: " + name);
}

}  // namespace nmtkit
