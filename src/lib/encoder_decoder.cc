#include <nmtkit/encoder_decoder.h>

#include <nmtkit/array.h>
#include <nmtkit/bidirectional_encoder.h>

/* Input/output mapping for training/force decoding:
 *
 *  Encoder Inputs                      Decoder Inputs
 * +-----------------------------+     +-----------------------------+
 *  s[0], s[1], s[2], ..., s[n-1]  |||  t[0], t[1], t[2], ..., t[m-2], t[m-1]
 *                                           +-------------------------------+
 *                                            Decoder Outputs
 */

using namespace std;

namespace DE = dynet::expr;

namespace nmtkit {

EncoderDecoder::EncoderDecoder(
    unsigned src_vocab_size,
    unsigned trg_vocab_size,
    unsigned embed_size,
    unsigned hidden_size,
    dynet::Model * model)
: rnn_dec_(1, embed_size, hidden_size, model) {
  encoder_.reset(
      new BidirectionalEncoder(
          1, src_vocab_size, embed_size, hidden_size, model));
  
  // Note: In this implementation, encoder and decoder are connected through one
  //       nonlinear intermediate embedding layer. The size of this layer is
  //       determined using the average of both modules.
  const unsigned enc_size = encoder_->getFinalStateSize();
  const unsigned dec_size = hidden_size;
  const unsigned ie_size = (enc_size + dec_size) / 2;

  enc2dec_.reset(
      new MultilayerPerceptron({enc_size, ie_size, dec_size}, model));
  dec2out_.reset(
      new MultilayerPerceptron({dec_size, trg_vocab_size}, model));

  p_dec_lookup_ = model->add_lookup_parameters(trg_vocab_size, {embed_size});
};

void EncoderDecoder::buildDecoderInitializerGraph(
    const DE::Expression & enc_final_state,
    dynet::ComputationGraph * cg,
    vector<DE::Expression> * dec_init_states) {
  // NOTE: LSTMBuilder::start_new_sequence() takes initial states with below
  //       layout:
  //         {c1, c2, ..., cn, h1, h2, ..., hn}
  //       where cx is the initial cell states and hx is the initial outputs.
  DE::Expression dec_init_c = enc2dec_->build(enc_final_state, cg);
  DE::Expression dec_init_h = DE::tanh(dec_init_c);
  *dec_init_states = {dec_init_c, dec_init_h};
}

void EncoderDecoder::buildDecoderGraph(
    const vector<DE::Expression> & dec_init_states,
    const vector<vector<unsigned>> & target_ids,
    dynet::ComputationGraph * cg,
    vector<DE::Expression> * dec_outputs) {
  dec_outputs->clear();
  const unsigned tl = target_ids.size() - 1;

  rnn_dec_.new_graph(*cg);
  rnn_dec_.start_new_sequence(dec_init_states);

  for (unsigned i = 0; i < tl; ++i) {
    DE::Expression embed = DE::lookup(*cg, p_dec_lookup_, target_ids[i]);
    DE::Expression dec_h = rnn_dec_.add_input(embed);
    DE::Expression dec_out = dec2out_->build(dec_h, cg);
    dec_outputs->emplace_back(dec_out);
  }
}

void EncoderDecoder::decodeForInference(
    const vector<DE::Expression> & dec_init_states,
    const unsigned bos_id,
    const unsigned eos_id,
    const unsigned max_length,
    dynet::ComputationGraph * cg,
    InferenceGraph * ig) {
  ig->clear();

  rnn_dec_.new_graph(*cg);
  rnn_dec_.start_new_sequence(dec_init_states);

  InferenceGraph::Node * prev_node = ig->addNode({bos_id, 0.0f});

  for (unsigned generated = 0; ; ++generated) {
    vector<unsigned> inputs {prev_node->label().word_id};
    DE::Expression embed = DE::lookup(*cg, p_dec_lookup_, inputs);
    DE::Expression dec_h = rnn_dec_.add_input(embed);
    DE::Expression dec_out = dec2out_->build(dec_h, cg);
    DE::Expression log_probs_expr = DE::log_softmax(dec_out);
    vector<dynet::real> log_probs = dynet::as_vector(
        cg->incremental_forward(log_probs_expr));

    unsigned output_id = eos_id;
    if (generated < max_length - 1) {
      output_id = Array::argmax(log_probs);
    }
    InferenceGraph::Node * next_node = ig->addNode(
        InferenceGraph::Label {
            output_id, static_cast<float>(log_probs[output_id])});
    ig->connect(prev_node, next_node);
    prev_node = next_node;
    if (output_id == eos_id) {
      break;
    }
  }
}

void EncoderDecoder::buildLossGraph(
    const vector<vector<unsigned>> & target_ids,
    const vector<DE::Expression> & dec_outputs,
    vector<DE::Expression> * losses) {
  losses->clear();
  const unsigned tl = target_ids.size() - 1;

  for (unsigned i = 0; i < tl; ++i) {
    DE::Expression loss = DE::pickneglogsoftmax(
        dec_outputs[i], target_ids[i + 1]);
    losses->emplace_back(loss);
  }
}

DE::Expression EncoderDecoder::buildTrainGraph(
    const Batch & batch,
    dynet::ComputationGraph * cg) {
  // Encode
  //vector<DE::Expression> enc_output_states;
  DE::Expression enc_final_state;
  //encoder_->build(batch.source_ids, cg, &enc_output_states, &enc_final_state);
  encoder_->build(batch.source_ids, cg, nullptr, &enc_final_state);

  // Initialize decoder
  vector<DE::Expression> dec_init_states;
  buildDecoderInitializerGraph(enc_final_state, cg, &dec_init_states);

  // Decode
  vector<DE::Expression> dec_outputs;
  buildDecoderGraph(dec_init_states, batch.target_ids, cg, &dec_outputs);

  // Calculate losses
  vector<DE::Expression> losses;
  buildLossGraph(batch.target_ids, dec_outputs, &losses);
  DE::Expression total_loss = DE::sum_batches(DE::sum(losses));
  return total_loss;
}

void EncoderDecoder::infer(
    const vector<unsigned> & source_ids,
    const unsigned bos_id,
    const unsigned eos_id,
    const unsigned max_length,
    dynet::ComputationGraph * cg,
    InferenceGraph * ig) {

  // Make batch data.
  vector<vector<unsigned>> source_ids_inner;
  source_ids_inner.emplace_back(vector<unsigned> {bos_id});
  for (const unsigned s : source_ids) {
    source_ids_inner.emplace_back(vector<unsigned> {s});
  }
  source_ids_inner.emplace_back(vector<unsigned> {eos_id});

  // Encode
  //vector<DE::Expression> enc_output_states;
  DE::Expression enc_final_state;
  //encoder_->build(batch.source_ids, cg, &enc_output_states, &enc_final_state);
  encoder_->build(source_ids_inner, cg, nullptr, &enc_final_state);

  // Initialize decoder
  vector<DE::Expression> dec_init_states;
  buildDecoderInitializerGraph(enc_final_state, cg, &dec_init_states);

  // Infer output words
  decodeForInference(dec_init_states, bos_id, eos_id, max_length, cg, ig);
}

}  // namespace nmtkit

NMTKIT_SERIALIZATION_IMPL(nmtkit::EncoderDecoder);

