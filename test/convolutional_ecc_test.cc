#include "config.h"

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <vector>
#include <nmtkit/convolutional_ecc.h>

using std::vector;
using nmtkit::ConvolutionalECC;

BOOST_AUTO_TEST_SUITE(ConvolutionalECCTest)

BOOST_AUTO_TEST_CASE(CheckNumBits) {
  for (unsigned r = 2; r <= 6; ++r) {
    ConvolutionalECC ecc(r);
    for (unsigned i = 0; i < 1024; ++i) {
      BOOST_CHECK_EQUAL(2 * (i + r), ecc.getNumBits(i));
    }
  }
}

BOOST_AUTO_TEST_CASE(CheckEncoding) {
  const vector<unsigned> num_registers {2, 3, 4, 5, 6};
  const vector<bool> input {0, 1, 0, 1, 0, 1};
  const vector<vector<bool>> expected {
    {0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1},
    {0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1},
    {0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1},
    {0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1},
    {0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1},
  };

  for (unsigned i = 0; i < num_registers.size(); ++i) {
    ConvolutionalECC ecc(num_registers[i]);
    const vector<bool> observed = ecc.encode(input);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected[i].begin(), expected[i].end(),
        observed.begin(), observed.end());
  }
}

BOOST_AUTO_TEST_CASE(CheckDecoding) {
  const float L = 0.45;
  const float EL = 0.45;
  const float H = 0.55;
  const float EH = 0.55;
  const vector<unsigned> num_registers {2, 3, 4, 5, 6};
  // Flips some bits to simulate errors.
  const vector<vector<float>> input {
    {L, L, EL, H, L, H, L, L, EH, H, L, L, L, H, H, H},
    {L, L, EL, H, H, H, H, L, L, EH, H, L, L, L, L, H, H, H},
    {L, L, EL, H, L, H, H, L, H, H, EH, H, H, H, H, L, H, L, H, H},
    {L, L, EL, H, L, H, L, L, L, L, H, EH, H, H, L, H, H, L, H, L, H, H},
    {L, L, EL, H, H, L, L, L, L, H, L, L, EH, L, L, L, H, L, H, H, L, H, H, H},
  };
  const vector<bool> expected {0, 1, 0, 1, 0, 1};

  for (unsigned i = 0; i < num_registers.size(); ++i) {
    ConvolutionalECC ecc(num_registers[i]);
    const vector<float> observed = ecc.decode(input[i]);
    vector<bool> observed_bits(observed.size());
    for (unsigned j = 0; j < observed.size(); ++j) {
      observed_bits[j] = observed[j] >= 0.5f;
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected.begin(), expected.end(),
        observed_bits.begin(), observed_bits.end());
  }
}

BOOST_AUTO_TEST_CASE(CheckEncDec) {
  const unsigned N = 64;
  const unsigned B = 6;
  const float L = 0.45;
  const float H = 0.55;

  for (const unsigned num_registers : {2, 3, 4, 5, 6}) {
    ConvolutionalECC ecc(num_registers);
    const unsigned num_encoded_bits = ecc.getNumBits(B);

    for (unsigned i = 0; i < N; ++i) {
      // Encode.
      vector<bool> input(B);
      for (unsigned j = 0; j < B; ++j) {
        input[j] = static_cast<bool>((i >> j) & 1);
        std::cout << input[j];
      }
      std::cout << std::endl;
      vector<bool> encoded = ecc.encode(input);

      // Flips a bit to insert an error.
      const unsigned pos = i % num_encoded_bits;
      encoded[pos] = !encoded[pos];

      // Decode.
      vector<float> prob(num_encoded_bits);
      for (unsigned j = 0; j < num_encoded_bits; ++j) {
        prob[j] = encoded[j] ? H : L;
      }
      const vector<float> observed = ecc.decode(prob);

      // Check observed bits.
      vector<bool> observed_bits(observed.size());
      for (unsigned j = 0; j < B; ++j) {
        observed_bits[j] = observed[j] >= 0.5f;
      }
      BOOST_CHECK_EQUAL_COLLECTIONS(
          input.begin(), input.end(),
          observed_bits.begin(), observed_bits.end());
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
