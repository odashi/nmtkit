#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/init.h>
#include <dynet/model.h>
#include <dynet/tensor.h>
#include <dynet/training.h>
#include <nmtkit/batch_converter.h>
#include <nmtkit/encoder_decoder.h>
#include <nmtkit/exception.h>
#include <nmtkit/inference_graph.h>
#include <nmtkit/monotone_sampler.h>
#include <nmtkit/sorted_random_sampler.h>
#include <nmtkit/vocabulary.h>

using namespace std;

namespace FS = boost::filesystem;
namespace PO = boost::program_options;
namespace PT = boost::property_tree;

namespace {

PO::variables_map parseArgs(int argc, char * argv[]) {
  PO::options_description opt_generic("Generic options");
  opt_generic.add_options()
    ("help", "Print this manual and exit.")
    ("config",
     PO::value<string>(),
     "(required) Location of the training configuration file.")
    ("model",
     PO::value<string>(),
     "(required) Location of the model directory.")
    ;

  PO::options_description opt;
  opt.add(opt_generic);

  // parse
  PO::variables_map args;
  PO::store(PO::parse_command_line(argc, argv, opt), args);
  PO::notify(args);

  // print usage
  if (args.count("help")) {
    cerr << "NMTKit trainer." << endl;
    cerr << "Author: Yusuke Oda (http://github.com/odashi/)" << endl;
    cerr << "Usage:" << endl;
    cerr << "  train --config CONFIG_FILE --model MODEL_DIRECTORY" << endl;
    cerr << "  train --help" << endl;
    cerr << opt << endl;
    exit(1);
  }

  // check required arguments
  const vector<string> required_args = {"config", "model"};
  bool ok = true;
  for (const string & arg : required_args) {
    if (!args.count(arg)) {
      cerr << "Missing argument: --" << arg << endl;
      ok = false;
    }
  }
  NMTKIT_CHECK(
      ok,
      "Some required arguments are missing.\n"
      "(--help to show all options)");

  return args;
}

void makeDirectory(const FS::path & dirpath) {
  NMTKIT_CHECK(
      !FS::exists(dirpath),
      "Directory or file already exists: " + dirpath.string());
  NMTKIT_CHECK(
      FS::create_directories(dirpath),
      "Could not create directory: " + dirpath.string());
}

template <class T>
void saveParameters(const FS::path & filepath, const T & obj) {
  std::ofstream ofs(filepath.string());
  NMTKIT_CHECK(
      ofs.is_open(), "Could not open file to write: " + filepath.string());
  boost::archive::text_oarchive oar(ofs);
  oar << obj;
}

void run(int argc, char * argv[]) try {
  // parse commandline args and config file.
  const auto args = ::parseArgs(argc, argv);

  // create model directory.
  FS::path model_dir(args["model"].as<string>());
  ::makeDirectory(model_dir);

  // copy and parse config file.
  FS::path cfg_filepath = model_dir / "config.ini";
  FS::copy_file(args["config"].as<string>(), cfg_filepath);
  PT::ptree config;
  PT::read_ini(cfg_filepath.string(), config);

  // create vocabularies.
  nmtkit::Vocabulary src_vocab(
      config.get<string>("Corpus.train_source"),
      config.get<unsigned>("Model.source_vocabulary"));
  nmtkit::Vocabulary trg_vocab(
      config.get<string>("Corpus.train_target"),
      config.get<unsigned>("Model.target_vocabulary"));
  src_vocab.save((model_dir / "source.vocab").string());
  trg_vocab.save((model_dir / "target.vocab").string());

  // "<s>" and "</s>" IDs
  //const unsigned bos_id = trg_vocab.getID("<s>");
  //const unsigned eos_id = trg_vocab.getID("</s>");

  // maximum lengths
  const unsigned train_max_length = config.get<unsigned>("Train.max_length");
  const unsigned test_max_length = 1024;

  // create samplers and batch converter.
  nmtkit::SortedRandomSampler train_sampler(
      config.get<string>("Corpus.train_source"),
      config.get<string>("Corpus.train_target"),
      src_vocab, trg_vocab, train_max_length,
      config.get<unsigned>("Train.num_words_in_batch"),
      config.get<unsigned>("Train.random_seed"));
  cout << "Loaded 'train' corpus." << endl;
  nmtkit::MonotoneSampler dev_sampler(
      config.get<string>("Corpus.dev_source"),
      config.get<string>("Corpus.dev_target"),
      src_vocab, trg_vocab, test_max_length, 1);
  cout << "Loaded 'dev' corpus." << endl;
  nmtkit::MonotoneSampler test_sampler(
      config.get<string>("Corpus.test_source"),
      config.get<string>("Corpus.test_target"),
      src_vocab, trg_vocab, test_max_length, 1);
  cout << "Loaded 'test' corpus." << endl;
  nmtkit::BatchConverter batch_converter(src_vocab, trg_vocab);

  // create new trainer and EncoderDecoder model.
  dynet::Model model;
  dynet::AdamTrainer trainer(
      &model,
      config.get<float>("Train.adam_alpha"),
      config.get<float>("Train.adam_beta1"),
      config.get<float>("Train.adam_beta2"),
      config.get<float>("Train.adam_eps"));
  cout << "Created new trainer." << endl;
  nmtkit::EncoderDecoder encdec(
      config.get<unsigned>("Model.source_vocabulary"),
      config.get<unsigned>("Model.target_vocabulary"),
      config.get<unsigned>("Model.embedding"),
      config.get<unsigned>("Model.rnn_hidden"),
      config.get<string>("Model.attention_type"),
      config.get<unsigned>("Model.attention_hidden"),
      &model);
  cout << "Created new encoder-decoder model." << endl;

  // Train/dev/test loop
  const unsigned max_iteration = config.get<unsigned>("Train.max_iteration");
  const unsigned eval_interval = config.get<unsigned>(
      "Train.evaluation_interval");
  unsigned long num_trained_samples = 0;
  float best_dev_log_ppl = 1e100;
  cout << "Start training." << endl;

  for (unsigned iteration = 1; iteration <= max_iteration; ++iteration) {
    // Training
    {
      vector<nmtkit::Sample> samples;
      train_sampler.getSamples(&samples);
      nmtkit::Batch batch;
      batch_converter.convert(samples, &batch);
      dynet::ComputationGraph cg;
      dynet::expr::Expression total_loss_expr = encdec.buildTrainGraph(
          batch, &cg);
      cg.forward(total_loss_expr);
      cg.backward(total_loss_expr);
      trainer.update();

      num_trained_samples += batch.source_ids[0].size();
      if (!train_sampler.hasSamples()) {
        train_sampler.rewind();
      }
    }

    if (iteration % eval_interval == 0) {
      float dev_log_ppl, test_log_ppl;

      // Devtest
      {
        unsigned num_outputs = 0;
        float total_loss = 0.0f;
        while (dev_sampler.hasSamples()) {
          vector<nmtkit::Sample> samples;
          dev_sampler.getSamples(&samples);
          nmtkit::Batch batch;
          batch_converter.convert(samples, &batch);
          dynet::ComputationGraph cg;
          dynet::expr::Expression total_loss_expr = encdec.buildTrainGraph(
              batch, &cg);
          num_outputs += batch.target_ids.size() - 1;
          total_loss += static_cast<float>(
              dynet::as_scalar(cg.forward(total_loss_expr)));
        }

        dev_log_ppl = total_loss / num_outputs;
        dev_sampler.rewind();
      }

      // Test
      {
        unsigned num_outputs = 0;
        float total_loss = 0.0f;
        while (test_sampler.hasSamples()) {
          vector<nmtkit::Sample> samples;
          test_sampler.getSamples(&samples);
          nmtkit::Batch batch;
          batch_converter.convert(samples, &batch);
          dynet::ComputationGraph cg;
          dynet::expr::Expression total_loss_expr = encdec.buildTrainGraph(
              batch, &cg);
          num_outputs += batch.target_ids.size() - 1;
          total_loss += static_cast<float>(
              dynet::as_scalar(cg.forward(total_loss_expr)));
        }

        test_log_ppl = total_loss / num_outputs;
        test_sampler.rewind();
      }

      const string fmt_str =
          "iteration=%d samples=%d dev-log-ppl=%.6e test-log-ppl=%.6e";
      const auto fmt = boost::format(fmt_str)
          % iteration % num_trained_samples % dev_log_ppl % test_log_ppl;
      cout << fmt.str() << endl;

      ::saveParameters(model_dir / "latest.trainer.params", trainer);
      ::saveParameters(model_dir / "latest.model.params", encdec);
      cout << "Saved 'latest' model." << endl;
      
      if (dev_log_ppl < best_dev_log_ppl) {
        best_dev_log_ppl = dev_log_ppl;
        ::saveParameters(
            model_dir / "best_dev_log_ppl.model.params", encdec);
        cout << "Saved 'best_dev_log_ppl' model." << endl;
      }
    }
  }
  cout << "Finished." << endl;
} catch (exception & ex) {
  cerr << ex.what() << endl;
  exit(1);
}

}  // namespace

int main(int argc, char * argv[]) {
  dynet::initialize(argc, argv);
  ::run(argc, argv);
  dynet::cleanup();
  return 0;
}
