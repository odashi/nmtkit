#!/bin/bash

find new_src -name \*.cc -or -name \*.h \
  | xargs submodules/google_styleguide/cpplint/cpplint.py \
  --extensions=cc,h \
  --filter=-legal/copyright \
  --linelength=80 \
  --root=new_src/include
