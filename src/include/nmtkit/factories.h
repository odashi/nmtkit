#ifndef NMTKIT_FACTORIES_H_
#define NMTKIT_FACTORIES_H_

#include <string>
#include <boost/shared_ptr.hpp>
#include <dynet/model.h>
#include <nmtkit/attention.h>
#include <nmtkit/decoder.h>
#include <nmtkit/encoder.h>

namespace nmtkit {

class Factory {
  Factory() = delete;
  Factory(const Factory &) = delete;
  Factory(Factory &&) = delete;
  Factory & operator=(const Factory &) = delete;
  Factory & operator=(Factory &&) = delete;

public:
  // Creates an Encoder object.
  //
  // Arguments:
  //   name: Identifier of the Encoder implementation.
  //         Available values:
  //           "backward": Backward RNN
  //           "bidirectional": Bidirectional RNN
  //           "forward": Forward RNN
  //   vocab_size: Vocabulary size of the input IDs.
  //   embed_size: Number of units in the input embedding layer.
  //   hidden_size: Number of units in the RNN hidden layer.
  //   model: Model object for training.
  //
  // Returns:
  //   A shared pointer of selected Encoder object.
  static boost::shared_ptr<Encoder> createEncoder(
      const std::string & name,
      const unsigned vocab_size,
      const unsigned embed_size,
      const unsigned hidden_size,
      dynet::Model * model);

  // Creates an Decoder object.
  //
  // Arguments:
  //   name: Identifier of the Decoder implementation.
  //         Available values:
  //           "default": Default RNN decoder
  //   vocab_size: Vocabulary size of the input IDs.
  //   in_embed_size: Number of units in the input embedding layer.
  //   out_embed_size: Number of units in the output embedding layer.
  //   hidden_size: Number of units in the RNN hidden layer.
  //   seed_size: Number of units in the seed layer.
  //   context_size: Number of units in the context layer.
  //   model: Model object for training.
  //
  // Returns:
  //   A shared pointer of selected Decoder object.
  static boost::shared_ptr<Decoder> createDecoder(
      const std::string & name,
      const unsigned vocab_size,
      const unsigned in_embed_size,
      const unsigned out_embed_size,
      const unsigned hidden_size,
      const unsigned seed_size,
      const unsigned context_size,
      dynet::Model * model);

  // Creates an Attention object.
  //
  // Arguments:
  //   name: Identifier of the Attention implementation.
  //         Available values:
  //           "mlp": Multilayer perceptron-based attention
  //                  (similar to [Bahdanau+14])
  //           "bilinear": Bilinear attention
  //                       ("general" method in [Luong+15])
  //   context_size: Number of units in all context inputs.
  //   controller_size: Number of units in the controller input.
  //   hidden_size: Number of units in the attention hidden layer.
  //                This value is used only in the "mlp" method.
  //   model: Model object for training.
  //
  // Returns:
  //   A shared pointer of selected Attention object.
  static boost::shared_ptr<Attention> createAttention(
      const std::string & name,
      const unsigned context_size,
      const unsigned controller_size,
      const unsigned hidden_size,
      dynet::Model * model);
};

}  // namespace nmtkit

#endif  // NMTKIT_FACTORIES_H_
