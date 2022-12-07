#!/usr/bin/bash

for SPLIT in "train" "dev" "test"; do
    for LANG in "en" "de"; do
        echo "Tokenizing ${SPLIT}-${LANG}"
            cat "data/CCrawl.de-en/${SPLIT}.${LANG}" \
                | sacremoses -j 20 -l ${LANG} tokenize \
                > "data/CCrawl.de-en/${SPLIT}.tok.${LANG}";
    done;
done;

# interleave both languages
paste -d '\n' data/CCrawl.de-en/train.{en,de} > data/CCrawl.de-en/train.all
paste -d '\n' data/CCrawl.de-en/train.tok.{en,de} > data/CCrawl.de-en/train.tok.all
