#! /usr/bin/env bash

set -euxo pipefail

for IMPUTE_TYPE in zero mean median knn; do
#for IMPUTE_TYPE in zero mice mean median knn; do
    for MODEL_TYPE in neural xgboost svm; do
        echo "--- Running ${MODEL_TYPE} with ${IMPUTE_TYPE} ---"
        pixi run PREFACE train \
            --samplesheet samplesheet.tsv \
            --outdir ./output/${MODEL_TYPE}_${IMPUTE_TYPE} \
            --model ${MODEL_TYPE} \
            --impute ${IMPUTE_TYPE} \
            --tune
    done
done
