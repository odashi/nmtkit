cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(nmtkit VERSION 0.1.0 LANGUAGES CXX)

# external libraries
add_subdirectory(submodules/dynet)
add_subdirectory(submodules/mteval)

#enable_testing()

# nmtkit sources
add_subdirectory(bin)
add_subdirectory(nmtkit)
#add_subdirectory(test)
