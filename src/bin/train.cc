#include "config.h"

#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/serialization/scoped_ptr.hpp>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/model.h>
#include <dynet/tensor.h>
#include <dynet/training.h>
#include <mteval/EvaluatorFactory.h>
#include <nmtkit/batch_converter.h>
#include <nmtkit/character_vocabulary.h>
#include <nmtkit/encoder_decoder.h>
#include <nmtkit/exception.h>
#include <nmtkit/factories.h>
#include <nmtkit/inference_graph.h>
#include <nmtkit/init.h>
#include <nmtkit/monotone_sampler.h>
#include <nmtkit/sorted_random_sampler.h>
#include <nmtkit/word_vocabulary.h>
#include <spdlog/spdlog.h>

using namespace std;

namespace FS = boost::filesystem;
namespace PO = boost::program_options;
namespace PT = boost::property_tree;

namespace {

// Analyzes commandline arguments.
//
// Arguments:
//   argc: Number of commandline arguments.
//   argv: Actual values of commandline arguments.
//
// Returns:
//   A key-value store of the analyzed commandline arguments.
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
    ("force", "Force to run the command regardless the amount of the memory.")
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

// Makes a new directory to given path.
//
// Arguments:
//   dirpath: Location to make a new directory.
void makeDirectory(const FS::path & dirpath) {
  NMTKIT_CHECK(
      !FS::exists(dirpath),
      "Directory or file already exists: " + dirpath.string());
  NMTKIT_CHECK(
      FS::create_directories(dirpath),
      "Could not create directory: " + dirpath.string());
}

// Initializes the global logger object.
//
// Arguments:
//   dirpath: Location of the directory to put log file.
//   log_level: Name of the logging level.
//   log_to_stderr: If true, the logger outputs the status to stderr as well as
//                  the log file. Otherwise, the logger only outputs to the log
//                  file.
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

// Initialize a vocabulary object using a corpus.
//
// Arguments:
//   corpus_filepath: Location of the corpus file to be analyzed.
//   vocab_type: Name of the vocabulary type.
//   vocab_size: Number of entries in the vocabulary.
//
// Returns:
//   A new pointer of the vocabulary object.
//
// TODO: This is a workaround for old Boost libraries. The function should
//       return a smart pointer, but boost::scoped_ptr is not movable, and the
//       serialization library does not support std::unique_ptr.
nmtkit::Vocabulary * createVocabulary(
    const string & corpus_filepath,
    const string & vocab_type,
    const unsigned vocab_size) {
  if (vocab_type == "character") {
    return new nmtkit::CharacterVocabulary(corpus_filepath, vocab_size);
  } else if (vocab_type == "word") {
    return new nmtkit::WordVocabulary(corpus_filepath, vocab_size);
  }
  NMTKIT_FATAL("Invalid vocabulary type: " + vocab_type);
}

// Initializes a trainer object.
//
// Arguments:
//   config: A config object generated by the config file.
//   model: A dynet::Model object to register the trainer object.
//
// Returns:
//   A new pointer of the trainer object.
//
// TODO: Ditto as createVocabulary().
dynet::Trainer * createTrainer(const PT::ptree & config, dynet::Model * model) {
  const string opt_type = config.get<string>("Train.optimizer_type");
  if (opt_type == "sgd") {
    return new dynet::SimpleSGDTrainer(
        *model,
        config.get<float>("Train.sgd_eta"));
  } else if (opt_type == "momentum") {
    return new dynet::MomentumSGDTrainer(
        *model,
        config.get<float>("Train.sgd_eta"),
        config.get<float>("Train.sgd_momentum"));
  } else if (opt_type == "adagrad") {
    return new dynet::AdagradTrainer(
        *model,
        config.get<float>("Train.adagrad_eta"),
        config.get<float>("Train.adagrad_eps"));
  } else if (opt_type == "adadelta") {
    return new dynet::AdadeltaTrainer(
        *model,
        config.get<float>("Train.adadelta_eps"),
        config.get<float>("Train.adadelta_rho"));
  } else if (opt_type == "adam") {
    return new dynet::AdamTrainer(
        *model,
        config.get<float>("Train.adam_alpha"),
        config.get<float>("Train.adam_beta1"),
        config.get<float>("Train.adam_beta2"),
        config.get<float>("Train.adam_eps"));
  }
  NMTKIT_FATAL("Invalid optimizer type: " + opt_type);
}

// Saves an serializable object.
//
// Arguments:
//   filepath: Location of the file to save the object.
//   archive_format: Name of the archive type.
//   obj: Target object.
template <class T>
void saveArchive(
    const FS::path & filepath,
    const string & archive_format,
    const T & obj) {
  ofstream ofs(filepath.string());
  NMTKIT_CHECK(
      ofs.is_open(), "Could not open file to write: " + filepath.string());
  if (archive_format == "binary") {
    boost::archive::binary_oarchive oar(ofs);
    oar << obj;
  } else if (archive_format == "text") {
    boost::archive::text_oarchive oar(ofs);
    oar << obj;
  } else {
    NMTKIT_FATAL("Invalid archive format: " + archive_format);
  }
}

// Calculates the log perplexity of given encoder-decoder model.
//
// Arguments:
//   encdec: Target encoder-decoder object.
//   sampler: Sampler object for the corpus to be evaluated.
//   converter: BatchConverter object to be used to convert samples.
//
// Returns:
//   The log perplexity score.
float evaluateLogPerplexity(
    nmtkit::EncoderDecoder & encdec,
    nmtkit::MonotoneSampler & sampler,
    nmtkit::BatchConverter & converter) {
  unsigned num_outputs = 0;
  float total_loss = 0.0f;
  sampler.rewind();
  while (sampler.hasSamples()) {
    vector<nmtkit::Sample> samples = sampler.getSamples();
    nmtkit::Batch batch;
    converter.convert(samples, &batch);
    dynet::ComputationGraph cg;
    dynet::expr::Expression total_loss_expr = encdec.buildTrainGraph(
        batch, 0.0, &cg);
    num_outputs += batch.target_ids.size() - 1;
    total_loss += static_cast<float>(
        dynet::as_scalar(cg.forward(total_loss_expr)));
  }
  return total_loss / num_outputs;
}

// Calculates the BLEU score of given encoder-decoder model.
//
// Arguments:
//   trg_vocab: Vocabulary object for the target language.
//   encdec: Target encoder-decoder object.
//   sampler: Sampler object for the corpus to be evaluated.
//   max_length: Maximum number of words in each hypothesis.
//
// Returns:
//   The BLEU score.
float evaluateBLEU(
    const nmtkit::Vocabulary & trg_vocab,
    nmtkit::EncoderDecoder & encdec,
    nmtkit::MonotoneSampler & sampler,
    const unsigned max_length) {
  const auto evaluator = MTEval::EvaluatorFactory::create("BLEU");
  const unsigned bos_id = trg_vocab.getID("<s>");
  const unsigned eos_id = trg_vocab.getID("</s>");
  MTEval::Statistics stats;
  sampler.rewind();
  while (sampler.hasSamples()) {
    vector<nmtkit::Sample> samples = sampler.getSamples();
    nmtkit::InferenceGraph ig = encdec.infer(
        samples[0].source, bos_id, eos_id, max_length, 1, 0.0f);
    const auto hyp_nodes = ig.findOneBestPath(bos_id, eos_id);
    vector<unsigned> hyp_ids;
    // Note: Ignore <s> and </s>.
    for (unsigned i = 1; i < hyp_nodes.size() - 1; ++i) {
      hyp_ids.emplace_back(hyp_nodes[i]->label().word_id);
    }
    MTEval::Sample eval_sample {hyp_ids, {samples[0].target}};
    stats += evaluator->map(eval_sample);
  }
  return evaluator->integrate(stats);
}

}  // namespace

int main(int argc, char * argv[]) {
  try {
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

    // Archive format to save models.
    const string archive_format = config.get<string>("Global.archive_format");

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
    global_config.force_run = !!args.count("force");
    nmtkit::initialize(global_config);

    // Creates vocabularies.
    boost::scoped_ptr<nmtkit::Vocabulary> src_vocab(
        ::createVocabulary(
            config.get<string>("Corpus.train_source"),
            config.get<string>("Model.source_vocabulary_type"),
            config.get<unsigned>("Model.source_vocabulary_size")));
    boost::scoped_ptr<nmtkit::Vocabulary> trg_vocab(
        ::createVocabulary(
            config.get<string>("Corpus.train_target"),
            config.get<string>("Model.target_vocabulary_type"),
            config.get<unsigned>("Model.target_vocabulary_size")));
    ::saveArchive(model_dir / "source.vocab", archive_format, src_vocab);
    ::saveArchive(model_dir / "target.vocab", archive_format, trg_vocab);

    // Maximum lengths
    const unsigned train_max_length = config.get<unsigned>("Batch.max_length");
    const unsigned test_max_length = 1024;
    const float train_max_length_ratio = config.get<float>(
        "Batch.max_length_ratio");
    const float test_max_length_ratio = 1e10;

    // Creates samplers and batch converter.
    nmtkit::SortedRandomSampler train_sampler(
        config.get<string>("Corpus.train_source"),
        config.get<string>("Corpus.train_target"),
        *src_vocab, *trg_vocab,
        config.get<string>("Batch.batch_method"),
        config.get<string>("Batch.sort_method"),
        config.get<unsigned>("Batch.batch_size"),
        train_max_length, train_max_length_ratio,
        config.get<unsigned>("Global.random_seed"));
    const int corpus_size = train_sampler.getCorpusSize();
    logger->info("Loaded 'train' corpus.");
    nmtkit::MonotoneSampler dev_sampler(
        config.get<string>("Corpus.dev_source"),
        config.get<string>("Corpus.dev_target"),
        *src_vocab, *trg_vocab, test_max_length, test_max_length_ratio, 1);
    logger->info("Loaded 'dev' corpus.");
    nmtkit::MonotoneSampler test_sampler(
        config.get<string>("Corpus.test_source"),
        config.get<string>("Corpus.test_target"),
        *src_vocab, *trg_vocab, test_max_length, test_max_length_ratio, 1);
    logger->info("Loaded 'test' corpus.");
    const auto fmt_corpus_size = boost::format(
        "Cleaned corpus size: train=%d dev=%d test=%d")
        % train_sampler.getCorpusSize() % dev_sampler.getCorpusSize() 
        % test_sampler.getCorpusSize();
    logger->info(fmt_corpus_size.str());
    nmtkit::BatchConverter batch_converter(*src_vocab, *trg_vocab);

    dynet::Model model;

    // Creates a new trainer.
    boost::scoped_ptr<dynet::Trainer> trainer(::createTrainer(config, &model));
    trainer->sparse_updates_enabled = false;
    logger->info("Created new trainer.");

    // Create a new encoder-decoder model.
    auto encoder = nmtkit::Factory::createEncoder(config, *src_vocab, &model);
    auto attention = nmtkit::Factory::createAttention(config, *encoder, &model);
    auto decoder = nmtkit::Factory::createDecoder(
        config, *trg_vocab, *encoder, &model);
    auto predictor = nmtkit::Factory::createPredictor(
        config, *trg_vocab, *decoder, &model);
    nmtkit::EncoderDecoder encdec(
        encoder, decoder, attention, predictor,
        config.get<string>("Train.loss_integration_type"));
    logger->info("Created new encoder-decoder model.");

    const string lr_decay_type = config.get<string>("Train.lr_decay_type");

    // Decaying factors
    float lr_decay = 1.0f;
    const float lr_decay_ratio = config.get<float>("Train.lr_decay_ratio");

    const float dropout_ratio = config.get<float>("Train.dropout_ratio");
    const unsigned max_iteration = config.get<unsigned>("Train.max_iteration");

    const string evaluation_type = config.get<string>("Train.evaluation_type");
    const unsigned eval_interval = config.get<unsigned>(
        "Train.evaluation_interval");
    unsigned long num_trained_samples = 0;
    unsigned long num_trained_words = 0;
    unsigned long next_eval_words = eval_interval;
    unsigned long next_eval_samples = eval_interval;
    if (evaluation_type == "epoch") {
      next_eval_samples = eval_interval * corpus_size;
    } else if (evaluation_type == "sample") {
      next_eval_samples = eval_interval;
    }
    auto next_eval_time = 
        std::chrono::system_clock::to_time_t(
        (std::chrono::system_clock::now() + std::chrono::minutes(eval_interval)));
    auto training_start_time = std::chrono::system_clock::now();
    auto epoch_start_time = std::chrono::system_clock::now();
    unsigned long next_epoch_samples = corpus_size;

    float best_dev_log_ppl = 1e100;
    float best_dev_bleu = -1e100;
    logger->info("Start training.");

    for (unsigned iteration = 1; iteration <= max_iteration; ++iteration) {
      // Training
      {
        vector<nmtkit::Sample> samples = train_sampler.getSamples();
        nmtkit::Batch batch;
        batch_converter.convert(samples, &batch);
        dynet::ComputationGraph cg;
        dynet::expr::Expression total_loss_expr = encdec.buildTrainGraph(
            batch, dropout_ratio, &cg);
        cg.forward(total_loss_expr);
        cg.backward(total_loss_expr);
        trainer->update(lr_decay);

        num_trained_samples += batch.source_ids[0].size();
        num_trained_words += batch.target_ids.size() * batch.target_ids[0].size();
        if (!train_sampler.hasSamples()) {
          train_sampler.rewind();
        }

        const string fmt_str = "Trained: batch=%d samples=%d current-proc-words=%d lr=%.6e";
        const auto fmt = boost::format(fmt_str)
            % iteration % num_trained_samples % num_trained_words % lr_decay;
        logger->info(fmt.str());
      }

      if (lr_decay_type == "batch") {
        lr_decay *= lr_decay_ratio;
      }

      if (num_trained_samples >= next_epoch_samples) {
        next_epoch_samples += corpus_size;
        auto elapsed_time = std::chrono::system_clock::now() - epoch_start_time;
        auto elapsed_time_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed_time).count();
        const auto fmt_epoch_time = boost::format(
                "Epoch finished: elapsed-time(sec)=%d") % elapsed_time_seconds;
        logger->info(fmt_epoch_time.str());
        epoch_start_time = std::chrono::system_clock::now();
      }

      if ((evaluation_type == "step" and iteration % eval_interval == 0) or
          (evaluation_type == "word" and num_trained_words >= next_eval_words) or
          (evaluation_type == "time" and 
           std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) >= next_eval_time) or
          ((evaluation_type == "epoch" or evaluation_type == "sample") and num_trained_samples >= next_eval_samples)) {
        next_eval_words += eval_interval;
        if (evaluation_type == "epoch") {
          next_eval_samples += eval_interval * corpus_size;
        } else if (evaluation_type == "sample") {
          next_eval_samples += eval_interval;
        }
        auto elapsed_time = std::chrono::system_clock::now() - training_start_time;
        auto elapsed_time_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed_time).count();
        auto eval_start_time = std::chrono::system_clock::now();

        logger->info("Evaluating...");

        const float dev_log_ppl = ::evaluateLogPerplexity(
            encdec, dev_sampler, batch_converter);
        const auto fmt_dev_log_ppl = boost::format(
            "Evaluated: batch=%d current-proc-words=%d elapsed-time(sec)=%d dev-log-ppl=%.6e")
            % iteration % num_trained_words % elapsed_time_seconds % dev_log_ppl;
        logger->info(fmt_dev_log_ppl.str());

        const float dev_bleu = ::evaluateBLEU(
            *trg_vocab, encdec, dev_sampler, train_max_length);
        const auto fmt_dev_bleu = boost::format(
            "Evaluated: batch=%d current-proc-words=%d elapsed-time(sec)=%d dev-bleu=%.6f")
            % iteration % num_trained_words % elapsed_time_seconds % dev_bleu;
        logger->info(fmt_dev_bleu.str());

        const float test_log_ppl = ::evaluateLogPerplexity(
            encdec, test_sampler, batch_converter);
        const auto fmt_test_log_ppl = boost::format(
            "Evaluated: batch=%d current-proc-words=%d elapsed-time(sec)=%d test-log-ppl=%.6e")
            % iteration % num_trained_words % elapsed_time_seconds % test_log_ppl;
        logger->info(fmt_test_log_ppl.str());

        const float test_bleu = ::evaluateBLEU(
            *trg_vocab, encdec, test_sampler, train_max_length);
        const auto fmt_test_bleu = boost::format(
            "Evaluated: batch=%d current-proc-words=%d elapsed-time(sec)=%d test-bleu=%.6f")
            % iteration % num_trained_words % elapsed_time_seconds % test_bleu;
        logger->info(fmt_test_bleu.str());

        if (lr_decay_type == "eval") {
          lr_decay *= lr_decay_ratio;
        }

        ::saveArchive(
            model_dir / "latest.trainer.params", archive_format, trainer);
        ::saveArchive(
            model_dir / "latest.model.params", archive_format, encdec);
        logger->info("Saved 'latest' model.");

        if (dev_log_ppl < best_dev_log_ppl) {
          best_dev_log_ppl = dev_log_ppl;
          FS::path trainer_path = model_dir / "best_dev_log_ppl.trainer.params";
          FS::path model_path = model_dir / "best_dev_log_ppl.model.params";
          FS::remove(trainer_path);
          FS::remove(model_path);
          FS::copy_file(model_dir / "latest.trainer.params", trainer_path);
          FS::copy_file(model_dir / "latest.model.params", model_path);
          logger->info("Saved 'best_dev_log_ppl' model.");
        } else {
          if (lr_decay_type == "logppl") {
            lr_decay *= lr_decay_ratio;
          }
        }

        if (dev_bleu > best_dev_bleu) {
          best_dev_bleu = dev_bleu;
          FS::path trainer_path = model_dir / "best_dev_bleu.trainer.params";
          FS::path model_path = model_dir / "best_dev_bleu.model.params";
          FS::remove(trainer_path);
          FS::remove(model_path);
          FS::copy_file(model_dir / "latest.trainer.params", trainer_path);
          FS::copy_file(model_dir / "latest.model.params", model_path);
          logger->info("Saved 'best_dev_bleu' model.");
        } else {
          if (lr_decay_type == "bleu") {
            lr_decay *= lr_decay_ratio;
          }
        }

        // not to include evaluation time in epoch elapsed time.
        auto eval_took_time = std::chrono::system_clock::now() - eval_start_time;
        epoch_start_time += eval_took_time;

        num_trained_words = 0;
        training_start_time = std::chrono::system_clock::now();
        next_eval_time = 
            std::chrono::system_clock::to_time_t(
            (std::chrono::system_clock::now() + std::chrono::minutes(eval_interval)));
      }
    }

    logger->info("Finished.");

  } catch (exception & ex) {
    cerr << ex.what() << endl;
    if (nmtkit::isInitialized()) {
      nmtkit::finalize();
    }
    exit(1);
  }

  // Finalizes all components.
  // Note: nmtkit::finalize() should not be placed in the try block, because
  //       some NMTKit/DyNet objects are not yet disposed before leaving the try
  //       block.
  nmtkit::finalize();
  return 0;
}

