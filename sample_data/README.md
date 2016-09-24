NMTKit Sample Data
==================

This directory includes a small parallel corpus for English-Japanese translation.
These data are extracted from [TANAKA Corpus](http://www.edrdg.org/wiki/index.php/Tanaka_Corpus).

English sentences are tokenized using [Stanford Tokenizer](http://nlp.stanford.edu/software/tokenizer.html) and lowercased.
Japanese sentences are tokenized using [KyTea](http://www.phontron.com/kytea/).

All texts are encoded in UTF-8. Sentence separator is `'\n'` and word separator is `' '`.

Corpus Statistics
-----------------

| File     | #sentences |  #words | #vocabulary |
|:---------|-----------:|--------:|------------:|
| train.en |     50,000 | 391,047 |       6,634 |
| train.ja |     50,000 | 565,618 |       8,774 |
| dev.en   |        500 |   3,931 |         816 |
| dev.ja   |        500 |   5,668 |         894 |
| test.en  |        500 |   3,998 |         839 |
| test.ja  |        500 |   5,635 |         884 |
