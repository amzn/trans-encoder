# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from torch.utils.data import DataLoader
import torch.nn.functional as F
from sentence_transformers import models, losses, util, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
import argparse
import logging 
import csv
import torch
import sys
import random
import tqdm
import gzip
import os
import pandas as pd
import numpy as np

from data import load_data

from sentence_transformers_ext.bi_encoder_eval import (
    EmbeddingSimilarityEvaluator, 
    EmbeddingSimilarityEvaluatorEnsemble,
    EmbeddingSimilarityEvaluatorAUC,
    EmbeddingSimilarityEvaluatorAUCEnsemble
)
from sentence_transformers_ext.cross_encoder_eval import (
    CECorrelationEvaluatorEnsemble, 
    CECorrelationEvaluatorAUC,
    CECorrelationEvaluatorAUCEnsemble
)


def eval_encoder(all_test, encoder, task="sts", enc_type="bi", ensemble=False):
    """
    Evaluate bi- or cross-encoders.
    Parameters
    ----------
        all_test: a dict of all test sets 
        encoder: a bi- or cross-enocder
        enc_type: a string specifying whether the encoder is a bi- or cross-encoder
        ensemble: a bool value indicating whether multiple encoders are used in input
    Returns
    ----------
        None
    """
    scores = []
    for name, data in all_test.items():
        if task in ["sts", "sickr", "sts_sickr", "custom"]:
            if enc_type == "bi":
                if ensemble:
                    test_evaluator = EmbeddingSimilarityEvaluatorEnsemble.from_input_examples(data, name=name)
                else:
                    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(data, name=name)
            elif enc_type == "cross":
                if ensemble:
                    test_evaluator = CECorrelationEvaluatorEnsemble.from_input_examples(data, name=name)
                else:
                    test_evaluator = CECorrelationEvaluator.from_input_examples(data, name=name)
            else:
                raise NotImplementedError()
        else:
            if enc_type == "bi":
                if ensemble:
                    test_evaluator = EmbeddingSimilarityEvaluatorAUCEnsemble.from_input_examples(data, name=name)
                else:
                    test_evaluator = EmbeddingSimilarityEvaluatorAUC.from_input_examples(data, name=name)
            elif enc_type == "cross":
                if ensemble:
                    test_evaluator = CECorrelationEvaluatorAUCEnsemble.from_input_examples(data, name=name)
                else:
                    test_evaluator = CECorrelationEvaluatorAUC.from_input_examples(data, name=name)
            else:
                raise NotImplementedError()
        scores += [test_evaluator(encoder)]
    logging.info (" & ".join(["%.2f" % (s*100) for s in scores]))
    logging.info (f"*****  test's avg spearman's rho: {sum(scores)/len(scores):.4f} ****")


def main():

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])


    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            default="princeton-nlp/unsup-simcse-bert-base-uncased",
            help="Transformers' model name or path")
    parser.add_argument("--task", type=str, default='sts')
    parser.add_argument("--mode", type=str, default='bi', help="cross|bi")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bi_encoder_pooling_mode", type=str,
            default='cls', help="cls|mean")

    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--model_name_or_path1", type=str, 
            default="princeton-nlp/unsup-simcse-bert-base-uncased")
    parser.add_argument("--model_name_or_path2", type=str, 
            default="princeton-nlp/unsup-simcse-roberta-base")
    parser.add_argument("--bi_encoder_pooling_mode1", type=str, default="cls")
    parser.add_argument("--bi_encoder_pooling_mode2", type=str, default="cls")
    parser.add_argument("--quick_test", action="store_true")

    args = parser.parse_args()
    print (args)

    ### read datasets
    all_pairs, all_test, dev_samples = load_data(args.task)

    if args.quick_test:
        all_pairs = all_pairs[:5000] # for quick testing

    print ("|raw sentence pairs|:", len(all_pairs))
    print ("|dev set|:", len(dev_samples))
    for key in all_test:
        print ("|test set: %s|" % key, len(all_test[key]))

    model_name = args.model_name_or_path 
    model_name1 = args.model_name_or_path1 
    model_name2 = args.model_name_or_path2 

    max_seq_length = 32
    device=args.device

    if not args.ensemble:
        logging.info ("########## load model and evaluate ##########")

        if args.mode == "bi":

            ###### Bi-encoder (sentence-transformers) ######
            logging.info(f"Loading bi-encoder model: {model_name}")
            # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
            word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

            # Apply mean pooling to get one fixed sized sentence vector
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=args.bi_encoder_pooling_mode)

            bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

            # eval bi-encoder
            logging.info ("Evaluate bi-encoder...")
            eval_encoder(all_test, bi_encoder, task=args.task, enc_type="bi")
    
            
        elif args.mode == "cross":

            ###### cross-encoder (sentence-transformers) ######
            logging.info(f"Loading cross-encoder model: {model_name}")

            cross_encoder = CrossEncoder(model_name, device=device)

            # eval cross-encoder
            logging.info ("Evaluate cross-encoder...")
            eval_encoder(all_test, cross_encoder, task=args.task, enc_type="cross")
            
        else:
            raise NotImplementedError() 

    else:
        logging.info ("########## load models and evaluate ##########")

        if args.mode == "bi":
            ###### Bi-encoder (sentence-transformers) ######
            logging.info(f"Loading bi-encoder1 model: {model_name1}")
            logging.info(f"Loading bi-encoder2 model: {model_name2}")
            
            # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
            word_embedding_model1 = models.Transformer(model_name1, max_seq_length=max_seq_length)
            word_embedding_model2 = models.Transformer(model_name2, max_seq_length=max_seq_length)

            # Apply mean pooling to get one fixed sized sentence vector
            pooling_model1 = models.Pooling(word_embedding_model1.get_word_embedding_dimension(), pooling_mode=args.bi_encoder_pooling_mode1)
            pooling_model2 = models.Pooling(word_embedding_model2.get_word_embedding_dimension(), pooling_mode=args.bi_encoder_pooling_mode2)

            bi_encoder1 = SentenceTransformer(modules=[word_embedding_model1, pooling_model1], device=device)
            bi_encoder2 = SentenceTransformer(modules=[word_embedding_model2, pooling_model2], device=device)

            # eval bi-encoder
            logging.info ("Evaluate bi-encoder (ensembled)...")
            eval_encoder(all_test, [bi_encoder1, bi_encoder2], task=args.task, enc_type="bi", ensemble=True)
            

        elif args.mode == "cross":

            ###### cross-encoder (sentence-transformers) ######
            logging.info(f"Loading cross-encoder1 model: {model_name1}")
            logging.info(f"Loading cross-encoder2 model: {model_name2}")

            cross_encoder1 = CrossEncoder(model_name1, device=device)
            cross_encoder2 = CrossEncoder(model_name2, device=device)
            
            # eval cross-encoder
            logging.info ("Evaluate cross-encoder (ensembled)...")
            eval_encoder(all_test, [cross_encoder1, cross_encoder2], task=args.task, enc_type="cross", ensemble=True)

        else:
            raise NotImplementedError() 

if __name__ == "__main__":
    main()
