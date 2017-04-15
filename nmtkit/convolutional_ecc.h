#ifndef NMTKIT_CONVOLUTIONAL_ECC_H_
#define NMTKIT_CONVOLUTIONAL_ECC_H_

#include <boost/serialization/base_object.hpp>
#include <nmtkit/error_correcting_code.h>

namespace nmtkit {

// Convolutional error correcting code.
class ConvolutionalECC : public ErrorCorrectingCode {
  ConvolutionalECC(const ConvolutionalECC &) = delete;
  ConvolutionalECC(ConvolutionalECC &&) = delete;
  ConvolutionalECC & operator=(const ConvolutionalECC &) = delete;
  ConvolutionalECC & operator=(ConvolutionalECC &&) = delete;

public:
  ConvolutionalECC() {}

  // Initializes convolutional code.
  //
  // Arguments:
  //   num_registers: Number of internal registers.
  ConvolutionalECC(const unsigned num_registers);

  ~ConvolutionalECC() override {}

  std::vector<bool> encode(
      const std::vector<bool> & original_bits) const override;
  std::vector<float> decode(
      const std::vector<float> & encoded_probs) const override;
  unsigned getNumBits(const unsigned num_original_bits) const override;

private:
  // Performs convolution.
  //
  // Arguments:
  //   rev_input: Order-reversed input bit sequences.
  //
  // Returns:
  //   Result of convolution of input and internal weights.
  std::vector<bool> convolute(const std::vector<bool> & rev_input) const;

  // Boost serialization interface.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & boost::serialization::base_object<ErrorCorrectingCode>(*this);
    ar & num_symbols_;
    ar & num_registers_;
  }

  unsigned num_symbols_;
  unsigned num_registers_;
};

}  // namespace nmtkit

NMTKIT_SERIALIZATION_DECL(nmtkit::ConvolutionalECC);

#endif  // NMTKIT_CONVOLUTIONAL_ECC_H_
