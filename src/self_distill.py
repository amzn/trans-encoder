# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sentence_transformers import models, losses, util, SentenceTransformer, LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
import argparse
import logging 
import sys
import random
import tqdm
import math
import os
import numpy as np
import pandas as pd

# import from local codes
from data import load_data
from eval import eval_encoder

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

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, 
        default="princeton-nlp/unsup-simcse-bert-base-uncased",
        help="Transformers' model name or path")
parser.add_argument("--task", type=str, default='sts')
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--cycle", type=int, default=3)
parser.add_argument("--bi_encoder_pooling_mode", type=str,
        default='cls', help="cls|mean")
parser.add_argument("--num_epochs_cross_encoder", type=int, default=1)
parser.add_argument("--num_epochs_bi_encoder", type=int, default=10)
parser.add_argument("--batch_size_cross_encoder", type=int, default=32)
parser.add_argument("--batch_size_bi_encoder", type=int, default=128)
parser.add_argument("--init_with_new_models", action="store_true")
parser.add_argument("--random_seed", type=int, default=2021)
#parser.add_argument("--use_raw_data_from_all_tasks", action="store_true")
parser.add_argument("--add_snli_data", type=int, default=0)
parser.add_argument("--custom_corpus_path", type=str, default=None)
parser.add_argument("--num_training_pairs", type=int, default=None)
parser.add_argument("--save_all_predictions", action="store_true")
parser.add_argument("--quick_test", action="store_true")


args = parser.parse_args()
print (args)

torch.manual_seed(args.random_seed)

#### Just some code to print debug information to stdout
logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


### read datasets
all_pairs, all_test, dev_samples = load_data(args.task, fpath=args.custom_corpus_path)

# load_pairs from other tasks
"""
if args.use_raw_data_from_all_tasks:
    all_pairs_qqp, _, _ = load_data("qqp")
    all_pairs_qnli, _, _ = load_data("qnli")
    all_pairs_mrpc, _, _ = load_data("mrpc")
    all_pairs = all_pairs + all_pairs_qqp + all_pairs_qnli + all_pairs_mrpc
"""

if args.add_snli_data != 0:
    random.seed(args.random_seed)
    all_pairs_snli, _, _ = load_data("snli")
    all_pairs_snli_sampled = random.sample(all_pairs_snli, args.add_snli_data)
    all_pairs = all_pairs + all_pairs_snli_sampled

if args.quick_test:
    all_pairs = all_pairs[:1000] # for quick testing

# randomly select training pairs for control study
if args.num_training_pairs is not None:
    print ("before sampling |raw sentence pairs|:", len(all_pairs))
    if args.num_training_pairs == -1:
        # use all
        pass
    else:    
        random.seed(args.random_seed)
        all_pairs = random.sample(all_pairs, args.num_training_pairs)

print ("|raw sentence pairs|:", len(all_pairs))
print ("|dev set|:", len(dev_samples))
for key in all_test:
    print ("|test set: %s|" % key, len(all_test[key]))

model_name = args.model_name_or_path #"princeton-nlp/unsup-simcse-bert-base-uncased"
simcse2base = {
    "princeton-nlp/unsup-simcse-roberta-base": "roberta-base", 
    "princeton-nlp/unsup-simcse-roberta-large": "roberta-large", 
    "princeton-nlp/unsup-simcse-bert-base-uncased": "bert-base-uncased",
    "princeton-nlp/unsup-simcse-bert-large-uncased": "bert-large-uncased"}
batch_size_cross_encoder = args.batch_size_cross_encoder
batch_size_bi_encoder = args.batch_size_bi_encoder
num_epochs_cross_encoder = args.num_epochs_cross_encoder 
num_epochs_bi_encoder = args.num_epochs_bi_encoder
max_seq_length = 32
total_cycle = args.cycle
device=args.device

logging.info ("########## load base model and evaluate ##########")

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

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

bi_encoder_dev_scores = []
cross_encoder_dev_scores = []
bi_encoder_path = model_name
all_predictions_bi_encoder = {}
all_predictions_cross_encoder = {}


for cycle in range(total_cycle):
    cycle += 1
    logging.info (f"########## cycle {cycle:.0f} starts ##########")

    ###### label data with bi-encoder ######
    # label sentence pairs with bi-encoder
    logging.info ("Label sentence pairs with bi-encoder...")

    # Two lists of sentences
    sents1 = [p[0] for p in all_pairs]
    sents2 = [p[1] for p in all_pairs]

    #Compute embedding for both lists
    embeddings1 = bi_encoder.encode(sents1, convert_to_tensor=True)
    embeddings2 = bi_encoder.encode(sents2, convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores = F.cosine_similarity(embeddings1, embeddings2)

    # save the predictions
    all_predictions_bi_encoder["bi_encoder_cycle_"+str(cycle-1)] = cosine_scores.cpu().numpy()

    # form (self-labelled) train set
    train_samples = []

    for i in range(len(sents1)):
        if args.task in ["qnli"]:
            train_samples.append(InputExample(texts=[sents1[i], sents2[i]], label=cosine_scores[i]))
        else:
            train_samples.append(InputExample(texts=[sents1[i], sents2[i]], label=cosine_scores[i]))
            train_samples.append(InputExample(texts=[sents2[i], sents1[i]], label=cosine_scores[i]))

    del bi_encoder, embeddings1, embeddings2, cosine_scores
    torch.cuda.empty_cache()

    ###### Cross-encoder learning ######
    if args.init_with_new_models:
        bi_encoder_path = simcse2base[model_name] #model_name # always use new model (PLM)
    logging.info(f"Loading cross-encoder model: {bi_encoder_path}")
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for cross-encoder model
    cross_encoder = CrossEncoder(bi_encoder_path, num_labels=1, device=device, max_length=64)

    # We wrap gold_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size_cross_encoder)

    # We add an evaluator, which evaluates the performance during training
    if args.task in ["sts", "sickr", "sts_sickr", "custom"]:
        evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='dev')
    else:
        evaluator = CECorrelationEvaluatorAUC.from_input_examples(dev_samples, name='dev')

    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs_cross_encoder * 0.1) #10% of train data for warm-up
    logging.info(f"Warmup-steps: {warmup_steps}")

    cross_encoder_path = f"output/cross-encoder/" \
        f"{args.task}_cycle{cycle}_{model_name.replace('/', '-')}-{start_time}" 

    # Train the cross-encoder model
    cross_encoder.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            evaluation_steps=200,
            use_amp=True,
            epochs=num_epochs_cross_encoder,
            warmup_steps=warmup_steps,
            output_path=cross_encoder_path)



    cross_encoder = CrossEncoder(cross_encoder_path, max_length=64, device=device)
    #cross_encoder = CrossEncoder(cross_encoder_path, device=device)

    # eval cross-encoder
    dev_score = evaluator(cross_encoder)
    cross_encoder_dev_scores.append(dev_score)
    logging.info (f"***** dev's spearman's rho: {dev_score:.4f} *****")

    ###### label data with cross-encoder ######
    # label sentence pairs with cross-encoder
    logging.info ("Label sentence pairs with cross-encoder...")
    silver_scores = cross_encoder.predict(all_pairs)
    silver_samples = list(InputExample(texts=[data[0], data[1]], label=score) for \
        data, score in zip(all_pairs, silver_scores))

    del cross_encoder
    torch.cuda.empty_cache()

    # save the predictions
    all_predictions_cross_encoder["cross_encoder_cycle_"+str(cycle)] = silver_scores

    ###### Bi-encoder learning ######
    if args.init_with_new_models:
        cross_encoder_path = model_name # always use new model (SimCSE)
    logging.info(f"Loading bi-encoder model: {cross_encoder_path}") 
    word_embedding_model = models.Transformer(cross_encoder_path, max_seq_length=32)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=args.bi_encoder_pooling_mode)
    bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

    train_dataloader = DataLoader(silver_samples, shuffle=True, batch_size=batch_size_bi_encoder)
    train_loss = losses.CosineSimilarityLoss(model=bi_encoder)

    if args.task in ["sts", "sickr", "sts_sickr", "custom"]:
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='dev')
    else:
        evaluator = EmbeddingSimilarityEvaluatorAUC.from_input_examples(dev_samples, name='dev')

    # Configure the training.
    #warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
    #logging.info(f"Warmup-steps: {warmup_steps}")

    bi_encoder_path = f"output/bi-encoder/" \
        f"{args.task}_cycle{cycle}_{model_name.replace('/', '-')}-{start_time}" 
    
    # Train the bi-encoder model
    bi_encoder.fit(
          train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs_bi_encoder,
          evaluation_steps=200,
          warmup_steps=0,
          output_path=bi_encoder_path,
          optimizer_params= {"lr": 5e-5},
          use_amp=True,
          )

    bi_encoder = SentenceTransformer(bi_encoder_path, device=device)

    # eval bi-encoder
    dev_score = evaluator(bi_encoder)
    bi_encoder_dev_scores.append(dev_score)
    logging.info (f"***** dev's spearman's rho: {dev_score:.4f} *****")

logging.info (bi_encoder_dev_scores)
logging.info (cross_encoder_dev_scores)

# best bi-encoder
best_cycle_bi_encoder = np.argmax(bi_encoder_dev_scores)+1
best_cycle_bi_encoder_path = f"output/bi-encoder/"\
    f"{args.task}_cycle{best_cycle_bi_encoder}_{model_name.replace('/', '-')}-{start_time}" 
# eval bi-encoder
logging.info (f"Evaluate best bi-encoder (from cycle {best_cycle_bi_encoder})...")
bi_encoder = SentenceTransformer(best_cycle_bi_encoder_path, device=device)
logging.info (best_cycle_bi_encoder_path)
eval_encoder(all_test, bi_encoder, task=args.task, enc_type="bi")


# best cross-encoder
best_cycle_cross_encoder = np.argmax(cross_encoder_dev_scores)+1
best_cycle_cross_encoder_path = f"output/cross-encoder/"\
    f"{args.task}_cycle{best_cycle_cross_encoder}_{model_name.replace('/', '-')}-{start_time}" 
# eval cross-encoder
logging.info (f"Evaluate best cross-encoder (from cycle {best_cycle_cross_encoder})...")
logging.info (best_cycle_cross_encoder_path)
#cross_encoder = CrossEncoder(best_cycle_cross_encoder_path, max_length=64, device=device)
cross_encoder = CrossEncoder(best_cycle_cross_encoder_path, device=device)
eval_encoder(all_test, cross_encoder, task=args.task, enc_type="cross")

# save all predictions
best_cycle_bi_encoder_csv_path = f"output/bi-encoder/" \
    f"{args.task}_cycle{best_cycle_bi_encoder}_{model_name.replace('/', '-')}-{start_time}_all_preds.csv" 
best_cycle_cross_encoder_csv_path = f"output/cross-encoder/" \
    f"{args.task}_cycle{best_cycle_cross_encoder}_{model_name.replace('/', '-')}-{start_time}_all_preds.csv"

if args.save_all_predictions:
    bi_pred_df = pd.DataFrame(np.array(list(all_predictions_bi_encoder.values())).T, columns=list(all_predictions_bi_encoder.keys()))
    cross_pred_df = pd.DataFrame(np.array(list(all_predictions_cross_encoder.values())).T, columns=list(all_predictions_cross_encoder.keys()))
    bi_pred_df.to_csv(best_cycle_bi_encoder_csv_path, index=False) 
    cross_pred_df.to_csv(best_cycle_cross_encoder_csv_path, index=False) 
    
logging.info ("\n")
print (args)
logging.info ("\n")
logging.info ("***** END *****")
