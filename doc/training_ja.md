翻訳モデルの学習
================


コーパスの準備
--------------

翻訳モデルを学習するには**対訳コーパス**が必要です。
これは異なる2言語で記述されたテキストファイルで、
行ごとに文の意味が対応している必要があります。
また日本語などの単語が連結した言語の場合、
事前に[MeCab](http://taku910.github.io/mecab/)などを用いて
**分かち書き**処理を行う必要があります。

対訳コーパスは以下の3種類に重複なく分割しておく必要があります。

<dl>
  <dt>train</dt><dd>主な学習データ。</dd>
  <dt>dev</dt><dd>学習結果の検証用データ。数百文程度</dd>
  <dt>test</dt><dd>学習結果の評価用データ。数百文程度</dd>
</dl>

NMTKitにはこれらの処理をあらかじめ行った
[サンプルデータ](https://github.com/odashi/nmtkit/tree/master/sample_data)
が付属しています。このデータは
[Tanaka Corpus](http://www.edrdg.org/wiki/index.php/Tanaka_Corpus)
から抽出した日英対訳文で、NMTKitの入力として適した形に整理してあります。


学習設定ファイルの準備
----------------------

学習を始める前に、どのような翻訳モデルを作成するかを記述した設定ファイルを
作成する必要があります。
どのコーパスを入力にするか、
ニューラルネットの大きさをどうするか、
といった各項目を設定ファイル中に全て記述します。

具体的な記述方法は
[記述済みのサンプル](https://github.com/odashi/nmtkit/blob/master/sample_data/sample_config.ini)
を参照して下さい。


学習
----


翻訳モデルの学習には**train**コマンドを使用します。
作成した設定ファイルの場所と、
学習結果の翻訳モデルを書き出す出力ディレクトリ名を指定します。

サンプルデータを用いて学習させる場合は以下の通り。

    src/bin/train \
        --config sample_data/sample_config.ini \
        --model model

trainコマンドは`--model`で指定した出力ディレクトリ以外には変更を加えません。
出力ディレクトリとして指定した場所に既に何かある場合はコマンドが失敗します。

DyNetはデフォルトで512MBのメモリを確保します。
メモリが足りなくて学習に失敗する場合は、
必要なメモリをMB単位で`--dynet-mem`オプションに指定します。
このオプションは`train`の直後に記述する必要があります。

    src/bin/train \
        --dynet-mem 4096 \
        --config sample_data/sample_config.ini \
        --model model

出力ディレクトリには以下のファイルが書き出されます。

<dl>
  <dt>config.ini</dt><dd>`--config`で指定した設定ファイルのコピー</dd>
  <dt>source.vocab</dt><dd>原言語の語彙一覧</dd>
  <dt>target.vocab</dt><dd>目的言語の語彙一覧</dd>
  <dt>*.model.params</dt><dd>翻訳モデルのパラメータ</dd>
  <dt>*.trainer.params</dt><dd>学習器のパラメータ</dd>
</dl>

paramsファイルの接頭辞には以下の意味があります。

<dl>
  <dt>latest</dt><dd>最も新しい学習結果</dd>
  <dt>best_dev_log_ppl</dt><dd>過去の学習結果のうち、検証用データで最も高い翻訳確率となるもの</dd>
</dl>

