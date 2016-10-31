How to Install NMTKit
=====================


Prerequisites for installing
----------------------------

NMTKit needs following libraries:

* **GNU C++** ... v4.9 or later (or other compatible compilers)
* **Boost C++ Library** ... v1.50 or later
* **Eigen** ... The newest development version
* **DyNet** ... v1.0-rc1.
* **CUDA** ... v7.5 or later


And the installation process requires following tools:

* **Git**
* **Mercurial**
* **autotools**.


Install Eigen
-------------

First we get the development version of Eigen using Mercurial.
This process could be done by only putting obtained files into an appropriate
location because Eigen consists of only header files:

    hg clone https://bitbucket.org/eigen/eigen/ /path/to/eigen


Install DyNet
-------------

Next we get and build DyNet:

    git clone https://github.com/clab/dynet.git /path/to/dynet
    cd /path/to/dynet
    mkdir build
    cd build

Case of using CPUs to calculate neural network:

    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j <threads>

Case of using CUDA:

    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DBACKEND=cuda
    make -j <threads>


Configuring library paths
-------------------------

All shared libraries of CUDA and DyNet should be visible from the NMTKit
binaries.
Add a configuration in your shell-rc file like:

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/cuda
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/dynet/build/dynet


Install NMTKit
--------------

    git clone https://github.com/odashi/nmtkit.git /path/to/nmtkit
    cd /path/to/nmtkit
    git submodule init
    git submodule update
    autoreconf -i

Case of using CPUs to calculate neural network:

    ./configure --with-eigen=/path/to/eigen --with-dynet=/path/to/dynet
    make

Case of using CUDA:

    ./configure --with-eigen=/path/to/eigen --with-dynet=/path/to/dynet --with-cuda=/path/to/cuda
    make


Validation
----------

    make check

Sample files could be used to validate the behavior of binaries:

    src/bin/train --config sample_data/tiny_config.ini --model model
    src/bin/decode --model model < sample_data/tiny.in
