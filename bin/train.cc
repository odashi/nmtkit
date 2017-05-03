#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
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
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/model.h>
#include <dynet/tensor.h>
#include <dynet/training.h>
#include <nmtkit/batch_converter.h>
#include <nmtkit/bleu_evaluator.h>
#include <nmtkit/encoder_decoder.h>
#include <nmtkit/exception.h>
#include <nmtkit/factories.h>
#include <nmtkit/inference_graph.h>
#include <nmtkit/init.h>
#include <nmtkit/logger.h>
#include <nmtkit/loss_evaluator.h>
#include <nmtkit/sorted_random_sampler.h>
#include <spdlog/spdlog.h>

using std::cerr;
using std::endl;
using std::exception;
using std::make_shared;
using std::map;
using std::ofstream;
using std::shared_ptr;
using std::string;
using std::vector;

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
    ("config",
     PO::value<vector<string>>()->multitoken(),
     "(required) Location of the training configuration file(s). "
     "If multiple files are specified, they are merged into one configuration object. "
     "Each property is overwritten by the latest definition.")
    ("model",
     PO::value<string>(),
     "(required) Location of the model directory.")
    ("force", "Force to run the command regardless the amount of the memory.")
    ("skip-saving",
     "Never save any model parameters to the file.")
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
    cerr << endl;
    cerr << "  (make an NMT model)" << endl;
    cerr << "  train --config CONFIG_FILE1 CONFIG_FILE2 ... \\\\" << endl;
    cerr << "        --model MODEL_DIRECTORY" << endl;
    cerr << endl;
    cerr << "  (show this manual)" << endl;
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

// Make a logger object.
//
// Arguments:
//   name: Name of the logger.
//   filepath: Location of the log file.
//   format: Formatting string.
//   enable_stderr: Whether or not to output string to stderr.
//
// Returns:
//   A pointer of a logger object.
shared_ptr<spdlog::logger> initializeLogger(
    const string & name,
    const string & filepath,
    const string & format,
    const bool enable_stderr) {
  vector<spdlog::sink_ptr> sinks;
  sinks.emplace_back(make_shared<spdlog::sinks::simple_file_sink_st>(filepath));
  if (enable_stderr) {
    sinks.emplace_back(make_shared<spdlog::sinks::stderr_sink_st>());
  }
  auto logger = make_shared<spdlog::logger>(name, begin(sinks), end(sinks));
  logger->set_pattern(format);
  logger->set_level(spdlog::level::trace);
  logger->flush_on(spdlog::level::trace);
  return logger;
}

// Merge multiple configurations into one property tree.
//
// Arguments:
//   cfg_filepaths: Locations to the configuration files.
//
// Returns:
//   An merged ptree object.
PT::ptree mergeConfigs(const vector<string> & cfg_filepaths) {
  PT::ptree cfg;
  for (const string path : cfg_filepaths) {
    PT::ptree additional;
    PT::read_ini(path, additional);

    std::function<void(
        const PT::ptree::path_type &, const PT::ptree &)> traverse = [&](
        const PT::ptree::path_type & child_path, const PT::ptree & child) {
      cfg.put(child_path, child.data());
      for (const auto gch : child) {
        const auto gch_path = child_path / PT::ptree::path_type(gch.first);
        traverse(gch_path, gch.second);
      }
    };
    traverse("", additional);
  }
  return cfg;
}

// Initializes a trainer object.
//
// Arguments:
//   config: A config object generated by the config file.
//   model: A dynet::Model object to register the trainer object.
//
// Returns:
//   A new pointer of the trainer object.
shared_ptr<dynet::Trainer> createTrainer(
    const PT::ptree & config,
    dynet::Model * model) {
  const string opt_type = config.get<string>("Train.optimizer_type");
  if (opt_type == "sgd") {
    return make_shared<dynet::SimpleSGDTrainer>(
        *model,
        config.get<float>("Train.sgd_eta"));
  } else if (opt_type == "momentum") {
    return make_shared<dynet::MomentumSGDTrainer>(
        *model,
        config.get<float>("Train.sgd_eta"),
        config.get<float>("Train.sgd_momentum"));
  } else if (opt_type == "adagrad") {
    return make_shared<dynet::AdagradTrainer>(
        *model,
        config.get<float>("Train.adagrad_eta"),
        config.get<float>("Train.adagrad_eps"));
  } else if (opt_type == "adadelta") {
    return make_shared<dynet::AdadeltaTrainer>(
        *model,
        config.get<float>("Train.adadelta_eps"),
        config.get<float>("Train.adadelta_rho"));
  } else if (opt_type == "adam") {
    return make_shared<dynet::AdamTrainer>(
        *model,
        config.get<float>("Train.adam_alpha"),
        config.get<float>("Train.adam_beta1"),
        config.get<float>("Train.adam_beta2"),
        config.get<float>("Train.adam_eps"));
  }
  NMTKIT_FATAL("Invalid optimizer type: " + opt_type);
}

// Saves a serializable object.
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

}  // namespace

int main(int argc, char * argv[]) {
  try {
    // Parses commandline args and the config file.
    const auto args = ::parseArgs(argc, argv);

    // Creates the model directory.
    FS::path model_dir(args["model"].as<string>());
    ::makeDirectory(model_dir);

    // Initializes the logger.
    auto glog = ::initializeLogger(
        "global",
        (model_dir / "training.log").string(),
        "[%Y-%m-%d %H:%M:%S.%e]\t[%l]\t%v",
        true);

    // Copies and parses the config file.
    FS::path cfg_filepath = model_dir / "config.ini";
    PT::ptree config = ::mergeConfigs(args["config"].as<vector<string>>());
    PT::write_ini(cfg_filepath.string(), config);

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
    global_config.weight_decay = config.get<float>("Train.weight_decay");
    nmtkit::initialize(global_config);

    // Creates vocabularies.
    shared_ptr<nmtkit::Vocabulary> src_vocab =
      nmtkit::Factory::createVocabulary(
          config.get<string>("Corpus.train_source"),
          config.get<string>("Model.source_vocabulary_type"),
          config.get<unsigned>("Model.source_vocabulary_size"));
    shared_ptr<nmtkit::Vocabulary> trg_vocab =
      nmtkit::Factory::createVocabulary(
          config.get<string>("Corpus.train_target"),
          config.get<string>("Model.target_vocabulary_type"),
          config.get<unsigned>("Model.target_vocabulary_size"));
    ::saveArchive(model_dir / "source.vocab", archive_format, src_vocab);
    ::saveArchive(model_dir / "target.vocab", archive_format, trg_vocab);

    // Maximum lengths
    const unsigned train_max_length = config.get<unsigned>("Batch.max_length");
    const float train_max_length_ratio = config.get<float>(
        "Batch.max_length_ratio");

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
    const unsigned corpus_size = train_sampler.getNumSamples();
    glog->info("Loaded training corpus.");

    const auto fmt_corpus_size = boost::format(
        "Cleaned train corpus size: %d") % train_sampler.getNumSamples();
    glog->info(fmt_corpus_size.str());

    // Creates evaluators.
    const string dev_src_file = config.get<string>("Corpus.dev_source");
    const string dev_trg_file = config.get<string>("Corpus.dev_target");
    const string test_src_file = config.get<string>("Corpus.test_source");
    const string test_trg_file = config.get<string>("Corpus.test_target");
    const map<string, shared_ptr<nmtkit::Evaluator>> evaluators = {
      { "dev-loss",
        make_shared<nmtkit::LossEvaluator>(
            dev_src_file, dev_trg_file, *src_vocab, *trg_vocab) },
      { "dev-bleu",
        make_shared<nmtkit::BLEUEvaluator>(
            dev_src_file, dev_trg_file, *src_vocab, *trg_vocab) },
      { "test-loss",
        make_shared<nmtkit::LossEvaluator>(
            test_src_file, test_trg_file, *src_vocab, *trg_vocab) },
      { "test-bleu",
        make_shared<nmtkit::BLEUEvaluator>(
            test_src_file, test_trg_file, *src_vocab, *trg_vocab) },
    };
    glog->info("Created evaluators.");

    // Creates score loggers.
    const map<string, shared_ptr<nmtkit::Logger>> loggers = {
      { "batch",
        make_shared<nmtkit::Logger>(
            (model_dir / "batch.log").string(), "%d") },
      { "dev-loss",
        make_shared<nmtkit::Logger>(
            (model_dir / "dev_loss.log").string(), "%.8e") },
      { "dev-bleu",
        make_shared<nmtkit::Logger>(
            (model_dir / "dev_bleu.log").string(), "%.8f") },
      { "test-loss",
        make_shared<nmtkit::Logger>(
            (model_dir / "test_loss.log").string(), "%.8e") },
      { "test-bleu",
        make_shared<nmtkit::Logger>(
            (model_dir / "test_bleu.log").string(), "%.8f") },
    };
    glog->info("Created loggers.");

    nmtkit::BatchConverter batch_converter(*src_vocab, *trg_vocab);
    dynet::Model model;

    // Creates a new trainer.
    shared_ptr<dynet::Trainer> trainer(::createTrainer(config, &model));
    // Disable sparse updates.
    // By default, DyNet udpates parameters over the only used values.
    // However, if we use sparse updates with a momentum optimization function
    // such as Adam, the updates are not strictly correct.
    // To deal with this problem, we disable sparse udpates function.
    // For more details, see
    // http://dynet.readthedocs.io/en/latest/unorthodox.html#sparse-updates
    trainer->sparse_updates_enabled = false;
    glog->info("Created new trainer.");

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
    glog->info("Created new encoder-decoder model.");

    // Gradient clipping
    const float gradient_clipping = config.get<float>("Train.gradient_clipping");
    NMTKIT_CHECK(
        gradient_clipping >= 0.0f,
        "gradient_clipping should be greater than 0.0");
    
    // NOTE(odashi):
    //   It is better to avoid a direct comparison of float equivalences.
    //   But this would work for now, because 0.0f might be able to be
    //   represented without errors.
    if (gradient_clipping != 0.0f) {
        trainer->clipping_enabled = true;
        trainer->clip_threshold = gradient_clipping;
    }

    const string lr_decay_type = config.get<string>("Train.lr_decay_type");

    // Decaying factors
    float lr_decay = 1.0f;
    const float lr_decay_ratio = config.get<float>("Train.lr_decay_ratio");
    const unsigned max_iteration = config.get<unsigned>("Train.max_iteration");
    const string eval_type = config.get<string>("Train.evaluation_type");
    const unsigned eval_interval = config.get<unsigned>(
        "Train.evaluation_interval");
    unsigned long num_trained_samples = 0;
    unsigned long num_trained_words = 0;
    unsigned long next_eval_words = eval_interval;
    unsigned long next_eval_samples = eval_interval;
    if (eval_type == "corpus") {
      next_eval_samples = eval_interval * corpus_size;
    } else if (eval_type == "sample") {
      next_eval_samples = eval_interval;
    }

    std::chrono::system_clock::time_point current_time = std::chrono::system_clock::now();
    auto next_eval_time =
        std::chrono::system_clock::to_time_t(
        (current_time + std::chrono::minutes(eval_interval)));
    std::chrono::system_clock::time_point training_start_time = current_time;
    std::chrono::system_clock::time_point epoch_start_time = current_time;

    const bool skip_saving = static_cast<bool>(args.count("skip-saving"));

    float best_dev_loss = 1e100;
    float best_dev_bleu = -1e100;
    glog->info("Start training.");

    for (unsigned iteration = 1; iteration <= max_iteration; ++iteration) {
      // Training
      {
        const vector<nmtkit::Sample> samples = train_sampler.getSamples();
        const nmtkit::Batch batch = batch_converter.convert(samples);
        dynet::ComputationGraph cg;
        dynet::expr::Expression total_loss_expr = encdec.buildTrainGraph(
            batch, &cg, true);
        cg.forward(total_loss_expr);
        cg.backward(total_loss_expr);
        trainer->update(lr_decay);

        num_trained_samples += batch.source_ids[0].size();
        num_trained_words +=  (batch.target_ids.size() - 1) * batch.target_ids[0].size();
        if (!train_sampler.hasSamples()) {
          train_sampler.rewind();
          const auto elapsed_time = std::chrono::system_clock::now() - epoch_start_time;
          const auto elapsed_time_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed_time).count();
          const auto fmt_corpus_time = boost::format(
                  "Corpus finished: elapsed-time(sec)=%d") % elapsed_time_seconds;
          glog->info(fmt_corpus_time.str());
          epoch_start_time = std::chrono::system_clock::now();
        }

        const string fmt_str = "Trained: batch=%d samples=%d words=%d lr=%.6e";
        const auto fmt = boost::format(fmt_str)
            % iteration % num_trained_samples % num_trained_words % lr_decay;
        glog->info(fmt.str());
      }

      if (lr_decay_type == "batch") {
        lr_decay *= lr_decay_ratio;
      }

      bool do_eval = false;
      do_eval = do_eval or (eval_type == "step" and iteration % eval_interval == 0);
      do_eval = do_eval or (eval_type == "word" and num_trained_words >= next_eval_words);
      do_eval = do_eval or (eval_type == "time" and
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) >= next_eval_time);
      do_eval = do_eval or ((eval_type == "corpus" or eval_type == "sample") and num_trained_samples >= next_eval_samples);

      if (do_eval){
        next_eval_words += eval_interval;
        if (eval_type == "corpus") {
          next_eval_samples += eval_interval * corpus_size;
        } else if (eval_type == "sample") {
          next_eval_samples += eval_interval;
        }
        auto elapsed_time = std::chrono::system_clock::now() - training_start_time;
        std::chrono::system_clock::time_point eval_start_time = std::chrono::system_clock::now();

        glog->info("Evaluating...");
        loggers.at("batch")->log(iteration);

        map<string, float> scores;
        for (const auto & kv : evaluators) {
          const float score = kv.second->evaluate(&encdec);
          scores[kv.first] = score;
          loggers.at(kv.first)->log(score);
          glog->info("Evaluated: " + kv.first);
        }

        if (lr_decay_type == "eval") {
          lr_decay *= lr_decay_ratio;
        }

        if (!skip_saving) {
          ::saveArchive(
              model_dir / "latest.trainer.params", archive_format, trainer);
          ::saveArchive(
              model_dir / "latest.model.params", archive_format, encdec);
          glog->info("Saved 'latest' model.");
        }

        if (scores.at("dev-loss") < best_dev_loss) {
          best_dev_loss = scores.at("dev-loss");
          if (!skip_saving) {
            FS::path trainer_path = model_dir / "best_dev_loss.trainer.params";
            FS::path model_path = model_dir / "best_dev_loss.model.params";
            FS::remove(trainer_path);
            FS::remove(model_path);
            FS::copy_file(model_dir / "latest.trainer.params", trainer_path);
            FS::copy_file(model_dir / "latest.model.params", model_path);
            glog->info("Saved 'best_dev_loss' model.");
          }
        } else {
          if (lr_decay_type == "loss") {
            lr_decay *= lr_decay_ratio;
          }
        }

        if (scores.at("dev-bleu") > best_dev_bleu) {
          best_dev_bleu = scores.at("dev-bleu");
          if (!skip_saving) {
            FS::path trainer_path = model_dir / "best_dev_bleu.trainer.params";
            FS::path model_path = model_dir / "best_dev_bleu.model.params";
            FS::remove(trainer_path);
            FS::remove(model_path);
            FS::copy_file(model_dir / "latest.trainer.params", trainer_path);
            FS::copy_file(model_dir / "latest.model.params", model_path);
            glog->info("Saved 'best_dev_bleu' model.");
          }
        } else {
          if (lr_decay_type == "bleu") {
            lr_decay *= lr_decay_ratio;
          }
        }

        // not to include evaluation time in epoch elapsed time.
        current_time = std::chrono::system_clock::now();
        auto eval_took_time = current_time - eval_start_time;
        epoch_start_time += eval_took_time;

        num_trained_words = 0;
        training_start_time = current_time;
        next_eval_time =
            std::chrono::system_clock::to_time_t(
            current_time + std::chrono::minutes(eval_interval));
      }
    }

    glog->info("Finished.");

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

