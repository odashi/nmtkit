How to Install NMTKit
=====================


Prerequisites for installing
----------------------------

NMTKit needs following libraries:

* **Boost C++ Library** ... v1.50 or later
* **Eigen** ... The newest development version
* **DyNet** ... v1.0-rc1.
* ~~**CUDA** ... v7.5 or later~~ *Currently not supported*.


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

Case of using CPUs to calculate neural network:

    git clone git@github.com:clab/dynet /path/to/dynet
    cd /path/to/dynet
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j <threads>
    

Configuring library paths
-------------------------

All shared libraries of DyNet should be visible from the NMTKit binaries.
Add a configuration in your shell-rc file like:

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/dynet/build/dynet


Install NMTKit
--------------

    git clone git@github.com:odashi/nmtkit /path/to/nmtkit
    cd /path/to/nmtkit
    git submodule init
    git submodule update
    autoreconf -i
    ./configure --with-eigen=/path/to/eigen --with-dynet=/path/to/dynet
    make


Validation
----------

    make check

Sample files could be used to validate the behavior of binaries:

    src/bin/train --config sample_data/tiny_config.ini --model model
    src/bin/decode --model model < sample_data/tiny.in
