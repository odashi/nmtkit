#include <iostream>
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
#include <dynet/init.h>
#include <nmtkit/corpus.h>
#include <nmtkit/encoder_decoder.h>
#include <nmtkit/exception.h>
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
    ;

  PO::options_description opt;
  opt.add(opt_generic);

  // parse
  PO::variables_map args;
  PO::store(PO::parse_command_line(argc, argv, opt), args);
  PO::notify(args);

  // print usage
  if (args.count("help")) {
    cerr << "NMTKit decoder." << endl;
    cerr << "Author: Yusuke Oda (http://github.com/odashi/)" << endl;
    cerr << "Usage:" << endl;
    cerr << "  decode --model MODEL_DIRECTORY" << endl;
    cerr << "         < INPUT_TOKENS" << endl;
    cerr << "         > OUTPUT_TOKENS" << endl;
    cerr << "  decode --help" << endl;
    cerr << opt << endl;
    exit(1);
  }

  // check required arguments
  const vector<string> required_args = {"model"};
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

template <class T>
void loadParameters(const FS::path & filepath, T * obj) {
  std::ifstream ifs(filepath.string());
  NMTKIT_CHECK(
      ifs.is_open(), "Could not open file to read: " + filepath.string());
  boost::archive::text_iarchive iar(ifs);
  iar >> *obj;
}

void run(int argc, char * argv[]) try {
  // parse commandline args.
  const auto args = ::parseArgs(argc, argv);

  FS::path model_dir(args["model"].as<string>());

  // parse config file.
  PT::ptree config;
  PT::read_ini((model_dir / "config.ini").string(), config);

  // load vocabularies.
  nmtkit::Vocabulary src_vocab((model_dir / "source.vocab").string());
  nmtkit::Vocabulary trg_vocab((model_dir / "target.vocab").string());
  
  // "<s>" and "</s>" IDs
  const unsigned bos_id = trg_vocab.getID("<s>");
  const unsigned eos_id = trg_vocab.getID("</s>");

  // maximum lengths
  const unsigned max_length = config.get<unsigned>("Train.max_length");

  // load EncoderDecoder model.
  nmtkit::EncoderDecoder encdec;
  ::loadParameters(model_dir / "latest.model.params", &encdec);

  // consume input lines and decode them.
  vector<string> input_words;
  while (nmtkit::Corpus::readTokens(&cin, &input_words)) {
    vector<unsigned> input_word_ids;
    nmtkit::Corpus::wordsToWordIDs(input_words, src_vocab, &input_word_ids);
    dynet::ComputationGraph cg;
    nmtkit::InferenceGraph ig;
    encdec.infer(input_word_ids, bos_id, eos_id, max_length, &cg, &ig);
    vector<const nmtkit::InferenceGraph::Node *> heads;
    ig.findNodes(&heads, [&](const nmtkit::InferenceGraph::Node & node) {
        return node.label().word_id == bos_id;
    });
    const nmtkit::InferenceGraph::Node * cur_node = heads[0];
    while (true) {
      cout << trg_vocab.getWord(cur_node->label().word_id) << ' '
           << cur_node->label().log_prob << endl;
      if (cur_node->next().size() == 0) {
        break;
      }
      cur_node = cur_node->next()[0];
    }
  }
} catch (exception & ex) {
  cerr << ex.what() << endl;
  exit(1);
}

}  // namespace

int main(int argc, char * argv[]) {
  dynet::initialize(argc, argv);
  ::run(argc, argv);
  dynet::cleanup();
}

