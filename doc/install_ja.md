NMTKitのインストール方法
========================


インストール要件
----------------

NMTKitは以下のライブラリに依存しています。

* **Boost C++ Library** ... v1.50 以降
* **Eigen** ... 最新の開発版
* **DyNet** ... v1.0-rc1
* ~~**CUDA** ... v7.5 以降~~ *現在未対応*

またインストール作業のために以下のツールが必要です。

* **Git**
* **Marcurial**
* **autotools**


Eigenのインストール
-------------------

MarcurialによりEigenの開発版を取得します。
Eigenはヘッダファイルのみで構成されるので、
適当なディレクトリに展開すれば終了です。

    hg clone https://bitbucket.org/eigen/eigen/ /path/to/eigen


DyNetのインストール
-------------------

DyNetをインストールします。

* ニューラルネットの計算にCPUを使用する場合

    git clone git@github.com:clab/dynet /path/to/dynet
    cd /path/to/dynet
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j <threads>


ライブラリパスの設定
--------------------

DyNetの共有ライブラリがNMTKitから見える場所に設置されている必要があります。
以下のようにパスを通す設定を各自のshell-rcなどに記述します。

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/dynet/build/dynet


NMTKitのインストール
--------------------

    git clone git@github.com:odashi/nmtkit /path/to/nmtkit
    cd /path/to/nmtkit
    git submodule init
    git submodule update
    autoreconf -i
    ./configure --with-eigen=/path/to/eigen --with-dynet=/path/to/dynet
    make


動作確認
--------

    make check

サンプルファイルを使って正常に動作するか確認できます。

    src/bin/train --config sample_data/tiny_config.ini --model model
    src/bin/decode --model model < sample_data/tiny.in
