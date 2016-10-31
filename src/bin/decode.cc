#include <iostream>
#include <memory>
#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
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
    ("format",
     PO::value<string>()->default_value("text"),
     "Output format.\n"
     "Available options:\n"
     "  text : One-best tokens in each line.\n"
     "  html : HTML document with detailed information.")
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
  const vector<string> required_args = {"format", "model"};
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
void loadParameters(const FS::path & filepath, T * obj) {
  std::ifstream ifs(filepath.string());
  NMTKIT_CHECK(
      ifs.is_open(), "Could not open file to read: " + filepath.string());
  boost::archive::text_iarchive iar(ifs);
  iar >> *obj;
}

void run(int argc, char * argv[]) try {
  // Parses commandline args.
  const auto args = ::parseArgs(argc, argv);

  FS::path model_dir(args["model"].as<string>());

  // Parses config file.
  PT::ptree config;
  PT::read_ini((model_dir / "config.ini").string(), config);

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

  // Retrieves the formatter.
  auto formatter = ::getFormatter(args["format"].as<string>());

  // Loads vocabularies.
  nmtkit::Vocabulary src_vocab((model_dir / "source.vocab").string());
  nmtkit::Vocabulary trg_vocab((model_dir / "target.vocab").string());
  
  // "<s>" and "</s>" IDs
  const unsigned bos_id = trg_vocab.getID("<s>");
  const unsigned eos_id = trg_vocab.getID("</s>");

  // Maximum generation length
  const unsigned max_length = config.get<unsigned>("Train.max_length");

  // Loads EncoderDecoder model.
  nmtkit::EncoderDecoder encdec;
  ::loadParameters(model_dir / "best_dev_log_ppl.model.params", &encdec);

  // Consumes input lines and decodes them.
  formatter->initialize(&cout);
  vector<string> input_words;
  while (nmtkit::Corpus::readTokens(&cin, &input_words)) {
    vector<unsigned> input_word_ids;
    nmtkit::Corpus::wordsToWordIDs(input_words, src_vocab, &input_word_ids);
    dynet::ComputationGraph cg;
    nmtkit::InferenceGraph ig;
    encdec.infer(input_word_ids, bos_id, eos_id, max_length, &cg, &ig);
    formatter->write(input_words, ig, src_vocab, trg_vocab, &cout);
  }

  // Finalizes all components.
  formatter->finalize(&cout);
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
