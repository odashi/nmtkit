#ifndef NMTKIT_FACTORIES_H_
#define NMTKIT_FACTORIES_H_

#include <memory>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <dynet/model.h>
#include <nmtkit/attention.h>
#include <nmtkit/decoder.h>
#include <nmtkit/encoder.h>
#include <nmtkit/predictor.h>
#include <nmtkit/vocabulary.h>

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
  //   config: ptree object with correctly-defined options.
  //   vocab: Vocabulary object for the source language.
  //   model: Model object for training.
  //
  // Returns:
  //   A shared pointer of selected Encoder object.
  static std::shared_ptr<Encoder> createEncoder(
      const boost::property_tree::ptree & config,
      const Vocabulary & vocab,
      dynet::Model * model);

  // Creates an Decoder object.
  //
  // Arguments:
  //   config: ptree object with correctly-defined options.
  //   vocab: Vocabulary object for the target langauge.
  //   encoder: Encoder object.
  //   model: Model object for training.
  //
  // Returns:
  //   A shared pointer of selected Decoder object.
  static std::shared_ptr<Decoder> createDecoder(
      const boost::property_tree::ptree & config,
      const Vocabulary & vocab,
      const Encoder & encoder,
      dynet::Model * model);

  // Creates an Attention object.
  //
  // Arguments:
  //   config: ptree object with correctly-defined options.
  //   encoder: Encoder object.
  //   model: Model object for training.
  //
  // Returns:
  //   A shared pointer of selected Attention object.
  static std::shared_ptr<Attention> createAttention(
      const boost::property_tree::ptree & config,
      const Encoder & encoder,
      dynet::Model * model);

  // Creates an Predictor object.
  //
  // Arguments:
  //   config: ptree object with correctly-defined options.
  //   vocab: Vocabulary object of the target language.
  //   decoder: Decoder object.
  //   model: Model object for training.
  //
  // Returns:
  //   A shared pointer of selected Predictor object.
  static std::shared_ptr<Predictor> createPredictor(
      const boost::property_tree::ptree & config,
      const Vocabulary & vocab,
      const Decoder & decoder,
      dynet::Model * model);
};

}  // namespace nmtkit

#endif  // NMTKIT_FACTORIES_H_
