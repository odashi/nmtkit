翻訳結果の生成
==============

[学習](https://github.com/odashi/nmtkit/tree/master/doc/training_ja.md)
により出力ディレクトリに`best_dev_log_ppl.model.params`が生成されていれば、
**decode**コマンドにより実際の翻訳結果を生成することができます。

    src/bin/decode \
        --model model \
        < submodules/small_parallel_enja/test.en \
        > result.ja

`decode`コマンドに渡す入力文は予め単語分割を行っておく必要があります。

`--format html`を指定することで、
翻訳中の様々な情報を記録したHTMLを出力することが可能です。

    src/bin/decode \
        --model model \
        --format html \
        < submodules/small_parallel_enja/test.en \
        > result.ja.html

どのような出力が得られるのかは、実際にサンプルデータを用いて生成した
[HTMLの例](https://github.com/odashi/nmtkit/tree/master/doc/test_top100.ja.html)
をご覧ください。

