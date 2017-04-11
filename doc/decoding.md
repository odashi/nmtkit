Generating translations using the trained model
===============================================

If we obtained the `best_dev_log_ppl.model.params` file in the output directory
by [training process](https://github.com/odashi/nmtkit/tree/master/doc/training.md),
we can generate output sentences using `decode` command:

    $ path/to/decode \
        --model model \
        < /path/to/nmtkit/submodules/small_parallel_enja/test.en \
        > result.ja

Note that the input data of the `decode` should be tokenized in advance.

If you want to generate HTML file with more detailed information during the
decoding process, use `--format html` option:

    $ /path/to/decode \
        --model model \
        --format html \
        < /path/to/nmtkit/submodules/small_parallel_enja/test.en \
        > result.ja.html

Here is the
[sample HTML output](https://odashi.github.io/nmtkit/doc/test_top100.ja.html)
in where you could see what kind of outputs are obtained.

