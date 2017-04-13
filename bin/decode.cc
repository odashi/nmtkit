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

using std::cerr;
using std::endl;
using std::string;
using std::vector;

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
    ("force", "Force to run the command regardless the amount of the memory.")
    ("force-decoding",
     "Forcely decode given reference texts.\n"
     "This flag also require specifying --reference.")
    ("beam-width",
     PO::value<unsigned>()->default_value(1),
     "Beam search width in the decoder inference.")
    ("word-penalty",
     PO::value<float>()->default_value(0.0f),
     "Positive bias value of the log word probability.")
    ("input",
     PO::value<string>(),
     "Location of the input text file. "
     "If this option is not set, STDIN would be used.")
    ("output",
     PO::value<string>(),
     "Location of the output file. "
     "If this option is not set, STDOUT would be used.")
    ("reference",
     PO::value<string>(),
     "Location of the reference text file.")
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
    cerr << endl;
    cerr << "  (one-best/beam search)" << endl;
    cerr << "  decode [options]" << endl;
    cerr << "         --model MODEL_DIRECTORY" << endl;
    cerr << "         < INPUT_FILE (--input INPUT_FILE)" << endl;
    cerr << "         > OUTPUT_FILE (--output OUTPUT_FILE)" << endl;
    cerr << endl;
    cerr << "  (force decoding)" << endl;
    cerr << "  decode [options]" << endl;
    cerr << "         --force-decoding" << endl;
    cerr << "         --model MODEL_DIRECTORY" << endl;
    cerr << "         --reference REFERENCE_FILE" << endl;
    cerr << "         < INPUT_FILE (--input INPUT_FILE)" << endl;
    cerr << "         > OUTPUT_FILE (--output OUTPUT_FILE)" << endl;
    cerr << endl;
    cerr << "  (show this manual)" << endl;
    cerr << "  decode --help" << endl;
    cerr << opt << endl;
    exit(1);
  }

  // Checks required arguments
  vector<string> required_args = {"format", "model", "beam-width"};
  if (args.count("force-decoding")) {
    required_args.emplace_back("reference");
  }

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

std::shared_ptr<nmtkit::Formatter> getFormatter(const std::string & name) {
  if (name == "text") {
    return std::shared_ptr<nmtkit::Formatter>(new nmtkit::SingleTextFormatter());
  } else if (name == "html") {
    return std::shared_ptr<nmtkit::Formatter>(new nmtkit::HTMLFormatter());
  }
  NMTKIT_FATAL("Unknown formatter name: " + name);
}

template <class T>
void loadArchive(
    const FS::path & filepath,
    const string & archive_format,
    T * obj) {
  std::ifstream ifs(filepath.string());
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

    // Open input/output/reference files.
    std::shared_ptr<std::istream> in_is, ref_is;
    std::shared_ptr<std::ostream> out_os;
    const bool has_reference = !!args.count("reference");

    if (!!args.count("input")) {
      const string filename = args["input"].as<string>();
      std::ifstream * ifs = new std::ifstream(filename);
      in_is.reset(ifs);
      NMTKIT_CHECK(
          ifs->is_open(), "Could not open an input file: " + filename);
    } else {
      in_is.reset(&std::cin, [](...){});
    }
    if (has_reference) {
      const string filename = args["reference"].as<string>();
      std::ifstream * ifs = new std::ifstream(filename);
      ref_is.reset(ifs);
      NMTKIT_CHECK(
          ifs->is_open(), "Could not open a reference file: " + filename);
    }
    if (!!args.count("output")) {
      const string filename = args["output"].as<string>();
      std::ofstream * ofs = new std::ofstream(filename);
      out_os.reset(ofs);
      NMTKIT_CHECK(
          ofs->is_open(), "Could not open an output file: " + filename);
    } else {
      out_os.reset(&std::cout, [](...){});
    }

    const bool force_decoding = !!args.count("force-decoding");

    formatter->initialize(out_os.get());

    // Consumes input lines and decodes them.
    string input_line;
    while (nmtkit::Corpus::readLine(in_is.get(), &input_line)) {
      // Convert input/reference texts into word IDs.
      const vector<unsigned> input_ids = src_vocab->convertToIDs(input_line);
      string ref_line;
      vector<unsigned> ref_ids;
      if (has_reference) {
        NMTKIT_CHECK(
            nmtkit::Corpus::readLine(ref_is.get(), &ref_line),
            "Could not read a next reference line.");
        ref_ids = trg_vocab->convertToIDs(ref_line);
      }

      // Obtain results.
      nmtkit::InferenceGraph ig =
          force_decoding ?
          encdec.forceDecode(
              input_ids, ref_ids, bos_id, eos_id) :
          encdec.infer(
              input_ids, bos_id, eos_id, max_length, beam_width, word_penalty);

      // Write results.
      formatter->write(
          input_line, ref_line, ig, *src_vocab, *trg_vocab, out_os.get());
    }

    formatter->finalize(out_os.get());

  } catch (std::exception & ex) {
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
