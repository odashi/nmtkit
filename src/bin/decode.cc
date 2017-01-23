#include "config.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/serialization/scoped_ptr.hpp>
#include <dynet/dynet.h>
#include <nmtkit/corpus.h>
#include <nmtkit/encoder_decoder.h>
#include <nmtkit/exception.h>
#include <nmtkit/html_formatter.h>
#include <nmtkit/init.h>
#include <nmtkit/single_text_formatter.h>
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
    ("model",
     PO::value<string>(),
     "(required) Location of the model directory.")
    ("model-prefix",
     PO::value<string>()->default_value("best_dev_log_ppl"),
     "Prefix of the model parameter file.")
    ("format",
     PO::value<string>()->default_value("text"),
     "Output format.\n"
     "Available options:\n"
     "  text : One-best tokens in each line.\n"
     "  html : HTML document with detailed information.")
    ("reference",
     PO::value<string>(),
     "Location of the reference text file.")
    ("force", "Force to run the command regardless the amount of the memory.")
    ("beam-width",
     PO::value<unsigned>()->default_value(1),
     "Beam search width in the decoder inference.")
    ("word-penalty",
     PO::value<float>()->default_value(0.0f),
     "Positive bias value of the log word probability.")
    ;

  PO::options_description opt;
  opt.add(opt_generic);

  // Parse
  PO::variables_map args;
  PO::store(PO::parse_command_line(argc, argv, opt), args);
  PO::notify(args);

  // Prints usage
  if (args.count("help")) {
    cerr << "NMTKit decoder." << endl;
    cerr << "Author: Yusuke Oda (http://github.com/odashi/)" << endl;
    cerr << "Usage:" << endl;
    cerr << "  decode [options]" << endl;
    cerr << "         --model MODEL_DIRECTORY" << endl;
    cerr << "         < INPUT_FILE" << endl;
    cerr << "         > OUTPUT_FILE" << endl;
    cerr << "  decode --help" << endl;
    cerr << opt << endl;
    exit(1);
  }

  // Checks required arguments
  const vector<string> required_args = {"format", "model", "beam-width"};
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

unique_ptr<nmtkit::Formatter> getFormatter(const std::string & name) {
  if (name == "text") {
    return unique_ptr<nmtkit::Formatter>(new nmtkit::SingleTextFormatter());
  } else if (name == "html") {
    return unique_ptr<nmtkit::Formatter>(new nmtkit::HTMLFormatter());
  }
  NMTKIT_FATAL("Unknown formatter name: " + name);
}

template <class T>
void loadArchive(
    const FS::path & filepath,
    const string & archive_format,
    T * obj) {
  ifstream ifs(filepath.string());
  NMTKIT_CHECK(
      ifs.is_open(), "Could not open file to read: " + filepath.string());
  if (archive_format == "binary") {
    boost::archive::binary_iarchive iar(ifs);
    iar >> *obj;
  } else if (archive_format == "text") {
    boost::archive::text_iarchive iar(ifs);
    iar >> *obj;
  }
}

}  // namespace

int main(int argc, char * argv[]) {
  try {
    // Parses commandline args.
    const auto args = ::parseArgs(argc, argv);

    FS::path model_dir(args["model"].as<string>());

    // Parses config file.
    PT::ptree config;
    PT::read_ini((model_dir / "config.ini").string(), config);

    // Archive format to load models.
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

    // Retrieves the formatter.
    auto formatter = ::getFormatter(args["format"].as<string>());

    // Loads vocabularies.
    boost::scoped_ptr<nmtkit::Vocabulary> src_vocab, trg_vocab;
    ::loadArchive(model_dir / "source.vocab", archive_format, &src_vocab);
    ::loadArchive(model_dir / "target.vocab", archive_format, &trg_vocab);

    // "<s>" and "</s>" IDs
    const unsigned bos_id = trg_vocab->getID("<s>");
    const unsigned eos_id = trg_vocab->getID("</s>");

    // Decoder settings
    const unsigned max_length = config.get<unsigned>("Batch.max_length");
    NMTKIT_CHECK(max_length > 0, "Batch.max_length should be greater than 0.");
    const unsigned beam_width = args["beam-width"].as<unsigned>();
    NMTKIT_CHECK(beam_width > 0, "beam-width should be greater than 0.");
    const float word_penalty = args["word-penalty"].as<float>();

    // Loads EncoderDecoder model.
    const string model_prefix = args["model-prefix"].as<string>();
    nmtkit::EncoderDecoder encdec;
    ::loadArchive(
        model_dir / (model_prefix + ".model.params"),
        archive_format, &encdec);

    formatter->initialize(&cout);

    // Loads reference texts.
    std::ifstream ref_ifs;
    if (!!args.count("reference")) {
      ref_ifs.open(args["reference"].as<string>());
      NMTKIT_CHECK(
          ref_ifs.is_open(), "Could not open the reference file to load: " + args["reference"].as<string>());
    }

    // Consumes input lines and decodes them.
    string input_line = "";
    string ref_line = "";
    while (nmtkit::Corpus::readLine(&cin, &input_line)) {
      if (ref_ifs.is_open()) {
        nmtkit::Corpus::readLine(&ref_ifs, &ref_line);
      }
      vector<unsigned> input_ids = src_vocab->convertToIDs(input_line);
      nmtkit::InferenceGraph ig = encdec.infer(
          input_ids, bos_id, eos_id, max_length, beam_width, word_penalty);
      formatter->write(input_line, ref_line, ig, *src_vocab, *trg_vocab, &cout);
    }

    formatter->finalize(&cout);

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
