#!/usr/bin/env bash

RUN_NAIVE=true
RUN_SBERT=false
RUN_BTOPIC=false
RUN_ARTM=false
RUN_TILING=false
RUN_SUM=false

ROOT_PATH="/data/shared/datasets/nlp/"

WIKI727K_FULL_PATH="${ROOT_PATH}wiki_727/"
WIKI727K_TRAIN_PATH="${ROOT_PATH}wiki_727/train/"
WIKI727K_TEST_PATH="${ROOT_PATH}wiki_727/test/"

WIKI50_PATH="../data/wiki_test_50/"

AMI_FULL_PATH="${ROOT_PATH}ami-corpus/topics/"  # in this folder there is no division on folders train and test, only files
AMI_PATH="${ROOT_PATH}ami-corpus/splits/"  # in this folder there are two subfolders: train and test
AMI_TRAIN_PATH="${ROOT_PATH}ami-corpus/splits/train/"
AMI_TEST_PATH="${ROOT_PATH}ami-corpus/splits/test/"

AMI_K_FOLDS_PATH="${ROOT_PATH}ami-corpus/ami_5_folds/"


if $RUN_NAIVE; then
    printf 'Running naive pipeline...\n'
    nice bash ./run_naive.sh $WIKI50_PATH wiki
    
    # nice bash ./run_naive.sh ${AMI_FULL_PATH} ami

    # for i in {1..5}
    # do
    #     nice bash ./run_naive.sh ${AMI_K_FOLDS_PATH}fold${i}/test ami
    # done

    # nice bash ./run_naive.sh /data/shared/datasets/nlp/wiki727test_embedded.hf/ wiki

    printf 'Naive pipeline is done.\n\n'
fi

if $RUN_SBERT; then
    printf 'Running sentence-bert pipeline...\n'
    # nice bash ./run_sbert.sh $WIKI50_PATH wiki
    CUDA_VISIBLE_DEVICES=0 nice bash ./run_sbert.sh $AMI_FULL_PATH ami
    printf 'Sentence-bert pipeline is done.\n\n'
fi

if $RUN_BTOPIC; then
    printf 'Running BERTopic pipeline...\n'
    # CUDA_VISIBLE_DEVICES=7 nice bash ./run_btopic.sh $AMI_TRAIN_PATH $AMI_TEST_PATH ami
    # CUDA_VISIBLE_DEVICES=7 nice bash ./run_btopic.sh ${AMI_K_FOLDS_PATH}fold1/train ${AMI_K_FOLDS_PATH}fold1/test ami

    for i in {1..5}
    do
        CUDA_VISIBLE_DEVICES=7 nice bash ./run_btopic.sh ${AMI_K_FOLDS_PATH}fold${i}/train ${AMI_K_FOLDS_PATH}fold${i}/test ami
    done

    # CUDA_VISIBLE_DEVICES=7 nice bash ./run_btopic.sh $AMI_FULL_PATH $AMI_FULL_PATH ami
    # CUDA_VISIBLE_DEVICES=6 nice python main.py btopic --train_path /data/shared/datasets/nlp/wiki727train_embedded.hf/ --test_path /data/shared/datasets/nlp/wiki727test_embedded.hf/ -dt wiki
    printf 'BERTopic pipeline is done.\n\n'
fi

if $RUN_ARTM; then
    printf 'Running BigARTM pipeline...\n'
    # nice bash ./run_artm.sh $AMI_PATH ami
    # for i in {1..5}
    # do
    #     nice bash ./run_artm.sh ${AMI_K_FOLDS_PATH}fold${i} ami
    # done

    nice bash ./run_artm.sh $WIKI727K_FULL_PATH wiki
    printf 'BigARTM pipeline is done.\n\n'
fi

if $RUN_TILING; then
    printf 'Running TopicTiling on sentence-bert embeddings pipeline...\n'
    nice bash ./run_tiling.sh $AMI_FULL_PATH ami
    # for i in {1..5}
    # do
    #     CUDA_VISIBLE_DEVICES=7 nice bash ./run_tiling.sh ${AMI_K_FOLDS_PATH}fold${i}/test ami
    # done

    # nice python main.py sbert scores -i /data/shared/datasets/nlp/wiki727test_embedded.hf/ -dt wiki
    printf 'TopicTiling pipeline is done.\n\n'
fi

if $RUN_SUM; then
    printf 'Running summarization pipeline...\n'
    # CUDA_VISIBLE_DEVICES=7 nice python main.py sumseg -i /data/shared/datasets/nlp/wiki_test_50/ -dt wiki
    # CUDA_VISIBLE_DEVICES=0 nice python main.py sumseg -i $AMI_FULL_PATH -dt ami
    CUDA_VISIBLE_DEVICES=0 nice python main.py sumseg -i /data/shared/datasets/nlp/wiki727test_summarized_1.hf -dt wiki
    printf 'Summarization pipeline is done.\n\n'
fi
