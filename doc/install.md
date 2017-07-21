How to Install NMTKit
=====================


Prerequisites for installing
----------------------------

NMTKit needs following libraries:

* **GNU C++** ... v4.9 or later (or other compatible compilers)
* **Boost C++ Library** ... v1.50 or later
* **[Eigen](http://eigen.tuxfamily.org/)** ... The newest development version
* **[DyNet](https://github.com/clab/dynet)** ... v1 series (v2 is not available)
* **[MTEval](https://github.com/odashi/mteval)** ... v1.0.0 or later
* **CUDA** ... v7.5 or later


And the installation process may require following tools:

* **Git**
* **Mercurial**
* **CMake** ... v3.1 or later


Install Eigen
-------------

First we get the development version of Eigen using Mercurial.
This process could be done by only putting obtained files into an appropriate
location because Eigen consists of only header files:

    $ hg clone https://bitbucket.org/eigen/eigen/ /path/to/eigen


Install DyNet
-------------

Next we get and build DyNet:

    $ git clone https://github.com/clab/dynet.git /path/to/dynet
    $ cd /path/to/dynet
    $ mkdir build
    $ cd build

Case of using CPUs to calculate neural network:

    $ cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    $ make -j <threads>

Case of using CUDA:

    $ cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DBACKEND=cuda
    $ make -j <threads>


Install MTEval
--------------

MTEval can be installed by similar way to DyNet:

    $ git clone https://github.com/odashi/mteval.git /path/to/mteval
    $ cd /path/to/mteval
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make -j <threads>


Install NMTKit
--------------

Prepare build tree for NMTKit:

    $ git clone https://github.com/odashi/nmtkit.git /path/to/nmtkit
    $ cd /path/to/nmtkit
    $ git submodule init
    $ git submodule update
    $ mkdir build
    $ cd build

And configure makefiles with all library locations:

    $ cmake .. \
        -DUSE_GPU=ON \              # need if you use CUDA
        -DCUDA_ROOT=/path/to/cuda \ # ditto
        -DEIGEN3_INCLUDE_DIR=/path/to/eigen \
        -DDYNET_INCLUDE_DIR=/path/to/dynet \
        -DDYNET_LIBRARY_DIR=/path/to/dynet/build/dynet \
        -DMTEVAL_INCLUDE_DIR=/path/to/mteval \
        -DMTEVAL_LIBRARY_DIR=/path/to/mteval/build/mteval

And then:

    $ make -j <threads>


Configuring library paths (optional)
------------------------------------

All linked libraries should be visible from the NMTKit frontend binaries.
To give library locations explicitly, for example, add configurations in your
shell-rc file like:

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/cuda/lib64
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/boost/lib
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/dynet/build/dynet
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/mteval/build/mteval
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/nmtkit/build/nmtkit


Validation
----------

    $ make test

Sample files could be used to validate the behavior of binaries:

    $ /path/to/nmtkit/build/bin/train \
        --config /path/to/nmtkit/sample_data/tiny_config.ini \
        --model model
    $ /path/to/nmtkit/build/bin/decode \
        --model model \
        < /path/to/nmtkit/sample_data/tiny.in
