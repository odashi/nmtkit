Training Translation Models
===========================


Preparing parallel corpus
-------------------------

To make translation models using NMTKit, the toolkit requires a **parallel
corpus**, which consists of two texts written in different languages,
and each lines of them have the same meaning each other.
For example, following texts in English:

    It works.
    The quick brown fox jumps over the lazy dog.
    Frankly, my dear, I don't give a damn.

and corresponding texts in French:

    Ça marche.
    Le vif renard brun saute par-dessus le chien paresseux.
    Franchement, ma chère, c'est le cadet de mes soucis.

could be assumed as a parallel corpus.

In addition, we usually need the **tokenization** preprocessing to separate
words and some additional symbols, such as comma (,), period (.), according to
the language to be trained.

Parallel corpus should be separated into 3 parts without overlapping each other:

<dl>
  <dt>train</dt><dd>Main training data to make the translation model (as many as possible to obtain good model).</dd>
  <dt>dev</dt><dd>Used in the parameter selecting (usually 500 to 5000 sentences).</dd>
  <dt>test</dt><dd>Used in the accuracy calculation (usually 500 to 5000 sentences).</dd>
</dl>

NMTKit has a
[small En/Ja corpus](https://github.com/odashi/small_parallel_enja)
as an submodule, which includes 50k pre-tokenized parallel sentences for the
convenience of the experiments.


Preparing training configuration script
---------------------------------------

Next we prepare the training configuration file, which includes all settings for
the training; location of the corpus files, network topology, and training
parameters.

For now, we can use the
[sample file](https://github.com/odashi/nmtkit/blob/master/sample_data/sample_config.ini)
to run the trainer using above 50k En/Ja corpus.
Of course users can specify their favorite settings by modifying parameters
according to the format of this configuration file.


Launching trainer
-----------------

We use the `train` command to launch the training process.
`train` requires 2 parameters: the location of the configuration script and
the location of the output directory.

To launch the trainer with the sample corpus, type following lines:

    $ /path/to/train \
        --config /path/to/nmtkit/sample_data/sample_config.ini \
        --model model

`train` command modifies only files in the output directory specified by the
`--model` option. `train` may fail when any problems occurd, for example, other
file or directory already exists in the path specified in `--model`.

Some files would be generated while training in the output directory:

<dl>
  <dt>config.ini</dt><dd>Copy of the configuration script specified by `--config`.</dd>
  <dt>training.log</dt><dd>Proceeding logs of the training process.</dd>
  <dt>source.vocab</dt><dd>List of vocabularies in the source language.</dd>
  <dt>target.vocab</dt><dd>List of vocabularies in the target language.</dd>
  <dt>*.model.params</dt><dd>Parameters of the translation model.</dd>
  <dt>*.trainer.params</dt><dd>Parameters of the trainer.</dd>
</dl>

The prefix of the `*.params` files means:

<dl>
  <dt>latest</dt><dd>The newest model of the traiing process.</dd>
  <dt>best_dev_log_ppl</dt><dd>The best model according to the translation probability over the *dev* data.</dd>
</dl>

We also could output similar contents as `training.log` to stderr by specifying
`--log-to-stderr` option:

    $ /path/to/train \
        --config /path/to/nmtkit/sample_data/sample_config.ini \
        --model model \
        --log-to-stderr
