翻訳結果の生成
==============

[学習](https://github.com/odashi/nmtkit/tree/master/doc/training_ja.md)
により出力ディレクトリにbest_dev_log_ppl.model.paramsが生成されていれば、
**decode**コマンドにより実際の翻訳結果を生成することができます。

    src/bin/decode \
        --model model \
        < submodules/small_parallel_enja/test.en \
        > result.ja

メモリが足りない場合は、
trainコマンドと同様に`--dynet-mem`オプションに適当な値を指定します。

    src/bin/decode \
        --dynet-mem 4096 \
        --model model \
        < submodules/small_parallel_enja/test.en \
        > result.ja


`--format html`を指定することで、
翻訳中の様々な情報を記録したHTMLを出力することが可能です。

    src/bin/decode \
        --model model \
        --format html \
        < submodules/small_parallel_enja/test.en \
        > result.ja.html

