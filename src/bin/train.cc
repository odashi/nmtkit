#include <fstream>
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
#include <dynet/model.h>
#include <dynet/tensor.h>
#include <dynet/training.h>
#include <nmtkit/batch_converter.h>
#include <nmtkit/encoder_decoder.h>
#include <nmtkit/exception.h>
#include <nmtkit/inference_graph.h>
#include <nmtkit/init.h>
#include <nmtkit/monotone_sampler.h>
#include <nmtkit/sorted_random_sampler.h>
#include <nmtkit/vocabulary.h>
#include <spdlog/spdlog.h>

using namespace std;

namespace FS = boost::filesystem;
namespace PO = boost::program_options;
namespace PT = boost::property_tree;

namespace {

PO::variables_map parseArgs(int argc, char * argv[]) {
  PO::options_description opt_generic("Generic options");
  opt_generic.add_options()
    ("help", "Print this manual and exit.")
    ("log-level",
     PO::value<string>()->default_value("info"),
     "Logging level to output.\n"
     "Available options:\n"
     "  trace (most frequent)\n"
     "  debug\n"
     "  info\n"
     "  warn\n"
     "  error\n"
     "  critical (fewest)")
    ("log-to-stderr",
     "Print logs to the stderr as well as the 'training.log' file.")
    ("config",
     PO::value<string>(),
     "(required) Location of the training configuration file.")
    ("model",
     PO::value<string>(),
     "(required) Location of the model directory.")
    ;

  PO::options_description opt;
  opt.add(opt_generic);

  // Parse
  PO::variables_map args;
  PO::store(PO::parse_command_line(argc, argv, opt), args);
  PO::notify(args);

  // Prints usage
  if (args.count("help")) {
    cerr << "NMTKit trainer." << endl;
    cerr << "Author: Yusuke Oda (http://github.com/odashi/)" << endl;
    cerr << "Usage:" << endl;
    cerr << "  train --config CONFIG_FILE --model MODEL_DIRECTORY" << endl;
    cerr << "  train --help" << endl;
    cerr << opt << endl;
    exit(1);
  }

  // Checks required arguments
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

void initializeLogger(
    const FS::path & dirpath,
    const string & log_level,
    bool log_to_stderr) {
  // Registers sinks.
  vector<spdlog::sink_ptr> sinks;
  sinks.emplace_back(
      std::make_shared<spdlog::sinks::simple_file_sink_st>(
          (dirpath / "training.log").string()));
  if (log_to_stderr) {
    sinks.emplace_back(std::make_shared<spdlog::sinks::stderr_sink_st>());
  }

  // Configures and registers the combined logger object.
  auto logger = std::make_shared<spdlog::logger>(
      "status", begin(sinks), end(sinks));
  logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e]\t[%l]\t%v");
  logger->flush_on(spdlog::level::trace);
  if (log_level == "trace") {
    logger->set_level(spdlog::level::trace);
  } else if (log_level == "debug") {
    logger->set_level(spdlog::level::debug);
  } else if (log_level == "info") {
    logger->set_level(spdlog::level::info);
  } else if (log_level == "warn") {
    logger->set_level(spdlog::level::warn);
  } else if (log_level == "error") {
    logger->set_level(spdlog::level::err);
  } else if (log_level == "critical") {
    logger->set_level(spdlog::level::critical);
  } else {
    NMTKIT_FATAL("Invalid log-level value: " + log_level);
  }
  spdlog::register_logger(logger);
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
  // Parses commandline args and the config file.
  const auto args = ::parseArgs(argc, argv);

  // Creates the model directory.
  FS::path model_dir(args["model"].as<string>());
  ::makeDirectory(model_dir);

  // Initializes the logger.
  ::initializeLogger(
      model_dir,
      args["log-level"].as<string>(),
      static_cast<bool>(args.count("log-to-stderr")));
  auto logger = spdlog::get("status");

  // Copies and parses the config file.
  FS::path cfg_filepath = model_dir / "config.ini";
  FS::copy_file(args["config"].as<string>(), cfg_filepath);
  PT::ptree config;
  PT::read_ini(cfg_filepath.string(), config);

  // Initializes NMTKit.
  nmtkit::GlobalConfig global_config;
  global_config.backend_random_seed = config.get<unsigned>(
      "Global.backend_random_seed");
  global_config.forward_memory_mb = config.get<unsigned>(
      "Global.forward_memory_mb");
  global_config.backward_memory_mb = config.get<unsigned>(
      "Global.backward_memory_mb");
  global_config.parameter_memory_mb = config.get<unsigned>(
      "Global.parameter_memory_mb");
  nmtkit::initialize(global_config);

  // Creates vocabularies.
  nmtkit::Vocabulary src_vocab(
      config.get<string>("Corpus.train_source"),
      config.get<unsigned>("Model.source_vocabulary"));
  nmtkit::Vocabulary trg_vocab(
      config.get<string>("Corpus.train_target"),
      config.get<unsigned>("Model.target_vocabulary"));
  src_vocab.save((model_dir / "source.vocab").string());
  trg_vocab.save((model_dir / "target.vocab").string());

  // Maximum lengths
  const unsigned train_max_length = config.get<unsigned>("Train.max_length");
  const unsigned test_max_length = 1024;
  const float train_max_length_ratio = config.get<float>(
      "Train.max_length_ratio");
  const float test_max_length_ratio = 1e10;

  // Creates samplers and batch converter.
  nmtkit::SortedRandomSampler train_sampler(
      config.get<string>("Corpus.train_source"),
      config.get<string>("Corpus.train_target"),
      src_vocab, trg_vocab, train_max_length, train_max_length_ratio,
      config.get<unsigned>("Train.num_words_in_batch"),
      config.get<unsigned>("Global.random_seed"));
  logger->info("Loaded 'train' corpus.");
  nmtkit::MonotoneSampler dev_sampler(
      config.get<string>("Corpus.dev_source"),
      config.get<string>("Corpus.dev_target"),
      src_vocab, trg_vocab, test_max_length, test_max_length_ratio, 1);
  logger->info("Loaded 'dev' corpus.");
  nmtkit::MonotoneSampler test_sampler(
      config.get<string>("Corpus.test_source"),
      config.get<string>("Corpus.test_target"),
      src_vocab, trg_vocab, test_max_length, test_max_length_ratio, 1);
  logger->info("Loaded 'test' corpus.");
  nmtkit::BatchConverter batch_converter(src_vocab, trg_vocab);

  // Creates new trainer and EncoderDecoder model.
  dynet::Model model;
  dynet::AdamTrainer trainer(
      &model,
      config.get<float>("Train.adam_alpha"),
      config.get<float>("Train.adam_beta1"),
      config.get<float>("Train.adam_beta2"),
      config.get<float>("Train.adam_eps"));
  logger->info("Created new trainer.");
  nmtkit::EncoderDecoder encdec(
      config.get<unsigned>("Model.source_vocabulary"),
      config.get<unsigned>("Model.target_vocabulary"),
      config.get<unsigned>("Model.embedding"),
      config.get<unsigned>("Model.rnn_hidden"),
      config.get<string>("Model.attention_type"),
      config.get<unsigned>("Model.attention_hidden"),
      &model);
  logger->info("Created new encoder-decoder model.");

  // Train/dev/test loop
  const unsigned max_iteration = config.get<unsigned>("Train.max_iteration");
  const unsigned eval_interval = config.get<unsigned>(
      "Train.evaluation_interval");
  unsigned long num_trained_samples = 0;
  float best_dev_log_ppl = 1e100;
  logger->info("Start training.");

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
      logger->info(fmt.str());

      ::saveParameters(model_dir / "latest.trainer.params", trainer);
      ::saveParameters(model_dir / "latest.model.params", encdec);
      logger->info("Saved 'latest' model.");
      
      if (dev_log_ppl < best_dev_log_ppl) {
        best_dev_log_ppl = dev_log_ppl;
        ::saveParameters(
            model_dir / "best_dev_log_ppl.model.params", encdec);
        logger->info("Saved 'best_dev_log_ppl' model.");
      }
    }
  }

  // Finalizes all components.
  logger->info("Finished.");
  nmtkit::finalize();

} catch (exception & ex) {
  cerr << ex.what() << endl;
  if (nmtkit::isInitialized()) {
    nmtkit::finalize();
  }
  exit(1);
}

}  // namespace

int main(int argc, char * argv[]) {
  ::run(argc, argv);
  return 0;
}
