#include <nmtkit/init.h>

#include <boost/format.hpp>
#include <dynet/init.h>
#include <nmtkit/exception.h>

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
