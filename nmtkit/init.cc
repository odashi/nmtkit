#include <config.h>

#include <nmtkit/init.h>

#include <iostream>
#if defined(__APPLE__) && defined(__MACH__)
#include <sys/sysctl.h>
#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>
#else
#include <sys/sysinfo.h>
#endif
#include <boost/format.hpp>
#include <dynet/init.h>
#include <nmtkit/exception.h>

using std::cerr;
using std::endl;
using std::vector;

namespace {

bool initialized = false;

// Check whether the requested memory size is fit to the actual memory or not.
//
// Arguments:
//   requested_size: Size of the requested memory in bytes.
void checkRequestedMemorySize(unsigned long requested_size) {
#if defined(__APPLE__) && defined(__MACH__)
  int mib[2];
  mib[0] = CTL_HW;
  mib[1] = HW_MEMSIZE;
  unsigned long total_ram = 0, free_ram = 0;
  size_t length = sizeof(unsigned long);
  sysctl(mib, 2, &total_ram, &length, NULL, 0);

  vm_size_t page_size;
  mach_port_t mach_port;
  mach_msg_type_number_t count;
  vm_statistics64_data_t vm_stats;

  mach_port = mach_host_self();
  count = sizeof(vm_stats) / sizeof(natural_t);
  if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
      KERN_SUCCESS == host_statistics64(mach_port, HOST_VM_INFO, (host_info64_t)&vm_stats, &count)) {
        free_ram = (unsigned long)vm_stats.free_count * (unsigned long)page_size;
  }
#else
  struct sysinfo info;
  if (::sysinfo(&info) != 0) {
    NMTKIT_FATAL("Something wrong in ::sysinfo().");
  }
  unsigned long total_ram = info.totalram * info.mem_unit;
  unsigned long free_ram = info.freeram * info.mem_unit;
#endif

  if (requested_size > total_ram * 2 / 3) {
    cerr << "Requested memory size exceeds the 2/3 of the total RAM:" << endl;
    cerr << "  Requested: " << (requested_size >> 20) << " MiB," << endl;
    cerr << "  Total    : " << (total_ram >> 20) << " MiB." << endl;
    cerr << "Please reconfirm the memory usage in your config script." << endl;
    cerr << "If you want to force to run the command with the current" << endl;
    cerr << "config, add `--force` as well as other arguments." << endl;
    NMTKIT_FATAL("Requested memory size exceeds the limit (2/3 of total).");
  }

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
  NMTKIT_CHECK(!::initialized, "NMTKit should not be initialized twice.");

  // Check memory size.
  if (!config.force_run) {
    const unsigned long total_memory_mb =
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
          % config.parameter_memory_mb
      ).str();
  params.weight_decay = config.weight_decay;
  params.shared_parameters = false;

#ifdef USE_GPU
  params.ngpus_requested = false;
  params.ids_requested = false;
  params.requested_gpus = -1;

  // Note: If the machine had 1025 or more GPUs then this process fails.
  const unsigned MAX_GPUS = 1024;
  params.gpu_mask = vector<int>(MAX_GPUS, 0);
#endif  // USE_GPU

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
