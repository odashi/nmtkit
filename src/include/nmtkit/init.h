#ifndef NMTKIT_INIT_H_
#define NMTKIT_INIT_H_

namespace nmtkit {

// Global configuration parameters of nmtkit.
struct GlobalConfig {
  // Seed value of the randomizer in the neural network backend (currently
  // DyNet).
  unsigned backend_random_seed;

  // Following 3 parameters specify the size of total reserved memory on the
  // selected device. The device should supports available memories at least
  // sum of their parameters.

  // Size of the reserved memory in MB for the forward path.
  unsigned forward_memory_mb;

  // Size of the reserved memory in MB for the backward path.
  // In most cases, this parameter could be set as same as forward_memory_mb.
  unsigned backward_memory_mb;

  // size of the reserved memory in MN for the shared parameters.
  unsigned parameter_memory_mb;
};

// Initializes NMTKit.
// This function should be called before all uses of NMTKit components.
//
// Arguments:
//   config: global configuration parameters.
void initialize(const GlobalConfig & config);

// Finalizes NMTKit.
// This function should be called just before exiting the program, i.e., all
// NMTKit/DyNet objects should be disposed before calling this function.
void finalize();

// Check whether NMTKit is initialized or not.
//
// Returns:
//   true if NMTKit is initialized, false otherwise.
bool isInitialized();

}  // namespace nmtkit

#endif  // NMTKIT_INIT_H_
