#include "config.h"

#include <nmtkit/init.h>

#include <boost/format.hpp>
#include <dynet/init.h>
#include <nmtkit/exception.h>

using namespace std;

namespace {

bool initialized = false;

}  // namespace

namespace nmtkit {

void initialize(const GlobalConfig & config) {
  NMTKIT_CHECK(!::initialized, "NMTKit should not be initialized twice.");

  dynet::DynetParams params;
  params.random_seed = config.backend_random_seed;
  params.mem_descriptor = (
      boost::format("%d,%d,%d")
          % config.forward_memory_mb
          % config.backward_memory_mb
          % config.parameter_memory_mb
      ).str();
  params.weight_decay = 0.0f;
  params.shared_parameters = false;

#if HAVE_CUDA
  params.ngpus_requested = false;
  params.ids_requested = false;
  params.requested_gpus = -1;

  // Note: If the machine had 1025 or more GPUs then this process fails.
  const unsigned MAX_GPUS = 1024;
  params.gpu_mask = vector<int>(MAX_GPUS, 0);
#endif  // HAVE_CUDA
  
  dynet::initialize(params);
  ::initialized = true;
}

void finalize() {
  NMTKIT_CHECK(::initialized, "NMTKit is not yet initialized.");
  dynet::cleanup();
  ::initialized = false;
}

bool isInitialized() {
  return ::initialized;
}

}  // namespace nmtkit
