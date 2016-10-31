Generating translations using the trained model
===============================================

If we obtained the `best_dev_log_ppl.model.params` file in the output directory
by [training process](https://github.com/odashi/nmtkit/tree/master/doc/training_ja.md),
we can generate output sentences using `decode` command:

    src/bin/decode \
        --model model \
        < submodules/small_parallel_enja/test.en \
        > result.ja

Note that the input data of the `decode` should be tokenized in advance.

If you want to generate HTML file with more detailed information during the
decoding process, use `--format html` option:

    src/bin/decode \
        --model model \
        --format html \
        < submodules/small_parallel_enja/test.en \
        > result.ja.html

Here is the
[sample HTML output](https://github.com/odashi/nmtkit/tree/master/doc/test_top100.ja.html)
in where you could see what kind of outputs are obtained.

