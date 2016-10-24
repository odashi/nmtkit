How to Install NMTKit
=====================


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

