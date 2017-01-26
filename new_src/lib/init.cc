#include <nmtkit/init.h>

#include <config.h>
#include <sys/sysinfo.h>
#include <boost/format.hpp>
#include <dynet/init.h>
#include <nmtkit/exception.h>
#include <iostream>

using std::cerr;
using std::endl;

namespace {

bool initialized = false;

// Check whether the requested memory size is fit to the actual memory or not.
//
// Arguments:
//   requested_size: Size of the requested memory in bytes.
void checkRequestedMemorySize(uint64_t requested_size) {
  struct sysinfo info;
  if (::sysinfo(&info) != 0) {
    NMTKIT_FATAL("Something wrong in ::sysinfo().");
  }

  const uint64_t total_ram = info.totalram * info.mem_unit;
  if (requested_size > total_ram * 2 / 3) {
    cerr << "Requested memory size exceeds the 2/3 of the total RAM:" << endl;
    cerr << "  Requested: " << (requested_size >> 20) << " MiB," << endl;
    cerr << "  Total    : " << (total_ram >> 20) << " MiB." << endl;
    cerr << "Please reconfirm the memory usage in your config script." << endl;
    cerr << "If you want to force to run the command with the current" << endl;
    cerr << "config, add `--force` as well as other arguments." << endl;
    NMTKIT_FATAL("Requested memory size exceeds the limit (2/3 of total).");
  }

  const uint64_t free_ram = info.freeram * info.mem_unit;
  if (requested_size > free_ram) {
    cerr << "Requested memory size exceeds the current free RAM:" << endl;
    cerr << "  Requested: " << (requested_size >> 20) << " MiB," << endl;
    cerr << "  Free     : " << (free_ram >> 20) << " MiB." << endl;
    cerr << "Please reconfirm the memory usage in your config script" << endl;
    cerr << "or refresh your machine." << endl;
    cerr << "If you want to force to run the command with the current" << endl;
    cerr << "config, add `--force` as well as other arguments." << endl;
    NMTKIT_FATAL("Requested memory size exceeds the free RAM.");
  }
}

}  // namespace

namespace nmtkit {

void initialize(const GlobalConfig & config) {
  NMTKIT_CHECK_MSG(!::initialized, "NMTKit should not be initialized twice.");

  // Check memory size.
  if (!config.force_run) {
    const uint64_t total_memory_mb =
        config.forward_memory_mb +
        config.backward_memory_mb +
        config.parameter_memory_mb;
    ::checkRequestedMemorySize(total_memory_mb << 20);
  }

  dynet::DynetParams params;
  params.random_seed = config.backend_random_seed;
  params.mem_descriptor = (
      boost::format("%d,%d,%d")
          % config.forward_memory_mb
          % config.backward_memory_mb
          % config.parameter_memory_mb).str();
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
  NMTKIT_CHECK_MSG(::initialized, "NMTKit is not yet initialized.");
  dynet::cleanup();
  ::initialized = false;
}

bool isInitialized() {
  return ::initialized;
}

}  // namespace nmtkit
