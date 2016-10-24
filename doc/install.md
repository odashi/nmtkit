How to Install NMTKit
=====================


Prerequisites for installing
----------------------------

NMTKit needs the latest version of **Boost C++ Library**, **Eigen**, **DyNet**.

The installation process requires **Git**, **Marcurial**, and **autotools**.


Install Eigen
-------------

Eigen is a C++ linear algebra toolkit which is used DyNet.

    hg clone https://bitbucket.org/eigen/eigen/ /path/to/eigen


Install DyNet
-------------

DyNet is a C++/Python neural network toolkit.

    git clone git@github.com:clab/dynet /path/to/dynet
    cd /path/to/dynet
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j <threads>
    

Configuring library paths
-------------------------

Add a configuration to locate DyNet shared library in your shell-rc file like:

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
    make check

