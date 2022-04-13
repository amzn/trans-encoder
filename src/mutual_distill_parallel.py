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

parser.add_argument("--task", type=str, default='sts',
        help='{sts|sickr|sts_sickr|qqp|qnli|mrpc|snli|custom}')
parser.add_argument("--device1", type=int, default=0)
parser.add_argument("--device2", type=int, default=0)
parser.add_argument("--cycle", type=int, default=3)
parser.add_argument("--num_epochs_cross_encoder", type=int, default=1)
parser.add_argument("--num_epochs_bi_encoder", type=int, default=10)
parser.add_argument("--batch_size_cross_encoder", type=int, default=32)
parser.add_argument("--batch_size_bi_encoder", type=int, default=128)
parser.add_argument("--init_with_new_models", action="store_true")
parser.add_argument("--use_large", action="store_true")
parser.add_argument("--bi_encoder1_pooling_mode", type=str,
        default='cls', help="{cls|mean}")
parser.add_argument("--bi_encoder2_pooling_mode", type=str,
        default='cls', help="{cls|mean}")
parser.add_argument("--random_seed", type=int, default=2021)
parser.add_argument("--custom_corpus_path", type=str, default=None)
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
if args.quick_test:
    all_pairs = all_pairs[:1000] # for quick test

print ("|raw sentence pairs|:", len(all_pairs))
print ("|dev set|:", len(dev_samples))
for key in all_test:
    print ("|test set: %s|" % key, len(all_test[key]))


if not args.use_large:
    model_name1 = "princeton-nlp/unsup-simcse-bert-base-uncased"
    model_name2 = "princeton-nlp/unsup-simcse-roberta-base" #'bert-base-uncased'
else:
    model_name1 = "princeton-nlp/unsup-simcse-bert-large-uncased"
    model_name2 = "princeton-nlp/unsup-simcse-roberta-large" #'bert-base-uncased'


simcse2base = {
    "princeton-nlp/unsup-simcse-roberta-base": "roberta-base", 
    "princeton-nlp/unsup-simcse-roberta-large": "roberta-large", 
    "princeton-nlp/unsup-simcse-bert-base-uncased": "bert-base-uncased",
    "princeton-nlp/unsup-simcse-bert-large-uncased": "bert-large-uncased"
}

batch_size_cross_encoder = args.batch_size_cross_encoder
batch_size_bi_encoder = args.batch_size_bi_encoder
num_epochs_cross_encoder = args.num_epochs_cross_encoder 
num_epochs_bi_encoder = args.num_epochs_bi_encoder
max_seq_length = 32
total_cycle = args.cycle
device1=args.device1
device2=args.device2

logging.info ("########## load base models and evaluate ##########")

###### Bi-encoder (sentence-transformers) ######
logging.info(f"Loading bi-encoder model1: {model_name1}")
logging.info(f"Loading bi-encoder model2: {model_name2}")
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model1 = models.Transformer(model_name1, max_seq_length=max_seq_length)
word_embedding_model2 = models.Transformer(model_name2, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model1 = models.Pooling(word_embedding_model1.get_word_embedding_dimension(), 
        pooling_mode=args.bi_encoder1_pooling_mode) # bert
pooling_model2 = models.Pooling(word_embedding_model1.get_word_embedding_dimension(), 
        pooling_mode=args.bi_encoder2_pooling_mode) # roberta

bi_encoder1 = SentenceTransformer(modules=[word_embedding_model1, pooling_model1], device=device1)
bi_encoder2 = SentenceTransformer(modules=[word_embedding_model2, pooling_model2], device=device2)

# eval bi-encoder
logging.info ("Evaluate bi-encoder (ensembled)...")
scores = []
for name, data in all_test.items():
    if args.task in ["sts", "sickr", "sts_sickr", "custom"]:
        test_evaluator = EmbeddingSimilarityEvaluatorEnsemble.from_input_examples(data, name=name)
    else:
        test_evaluator = EmbeddingSimilarityEvaluatorAUCEnsemble.from_input_examples(data, name=name)
    scores += [test_evaluator([bi_encoder1, bi_encoder2])]
logging.info (f"*****  test's avg spearman's rho: {sum(scores)/len(scores):.4f} ****")

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

bi_encoder_dev_scores = []
cross_encoder_dev_scores = []

for cycle in range(1, total_cycle+1):
    logging.info (f"########## cycle {cycle:.0f} starts ##########")
    
    ###### label data with bi-encoder ######
    # label sentence pairs with bi-encoder
    logging.info ("Label sentence pairs...")

    # Two lists of sentences
    sents1 = [p[0] for p in all_pairs]
    sents2 = [p[1] for p in all_pairs]

    #Compute embedding for both lists
    embeddings1 = bi_encoder1.encode(sents1, convert_to_tensor=True)
    embeddings2 = bi_encoder1.encode(sents2, convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores1 = F.cosine_similarity(embeddings1, embeddings2)

    embeddings1 = bi_encoder2.encode(sents1, convert_to_tensor=True)
    embeddings2 = bi_encoder2.encode(sents2, convert_to_tensor=True)
    cosine_scores2 = F.cosine_similarity(embeddings1, embeddings2)
    
    cosine_scores = torch.stack([cosine_scores1.cpu(), cosine_scores2.cpu()]).mean(0)

    # form (self-labelled) train set
    train_samples = []

    for i in range(len(sents1)):
        if args.task in ["qnli"]:
            train_samples.append(InputExample(texts=[sents1[i], sents2[i]], label=cosine_scores[i]))
        else:
            train_samples.append(InputExample(texts=[sents1[i], sents2[i]], label=cosine_scores[i]))
            train_samples.append(InputExample(texts=[sents2[i], sents1[i]], label=cosine_scores[i]))
    
    del bi_encoder1, bi_encoder2, embeddings1, embeddings2, cosine_scores1, cosine_scores2
    torch.cuda.empty_cache()

    ###### Cross-encoder learning ######
    logging.info(f"Loading cross-encoder1 model: {simcse2base[model_name1]}")
    logging.info(f"Loading cross-encoder2 model: {simcse2base[model_name2]}")
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for cross-encoder model
    cross_encoder1 = CrossEncoder(simcse2base[model_name1], num_labels=1, device=device1, max_length=64)
    cross_encoder2 = CrossEncoder(simcse2base[model_name2], num_labels=1, device=device2, max_length=64)

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

    cross_encoder_path1 = f"output/cross-encoder/" \
        f"{args.task}_cycle{cycle}_mutual_parallel_{model_name1.replace('/', '-')}-{start_time}" 
    cross_encoder_path2 = f"output/cross-encoder/" \
        f"{args.task}_cycle{cycle}_mutual_parallel_{model_name2.replace('/', '-')}-{start_time}"


    # Train the cross-encoder model
    cross_encoder1.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            evaluation_steps=200,
            use_amp=True,
            epochs=num_epochs_cross_encoder,
            warmup_steps=warmup_steps,
            output_path=cross_encoder_path1)

    # Train the cross-encoder model
    cross_encoder2.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            evaluation_steps=200,
            use_amp=True,
            epochs=num_epochs_cross_encoder,
            warmup_steps=warmup_steps,
            output_path=cross_encoder_path2)

    
    cross_encoder1 = CrossEncoder(cross_encoder_path1, max_length=64, device=device1)
    cross_encoder2 = CrossEncoder(cross_encoder_path2, max_length=64, device=device2)

    dev_score1 = evaluator(cross_encoder1)
    dev_score2 = evaluator(cross_encoder2)
    cross_encoder_dev_scores.append([dev_score1, dev_score2])
    logging.info (f"***** dev's spearman's rho: cross-encoder1 {dev_score1:.4f}, cross-encoder2 {dev_score2:.4f} *****")

    ###### label data with cross-encoder ######
    # label sentence pairs with cross-encoder
    logging.info ("Label sentence pairs...")
    silver_scores1 = cross_encoder1.predict(all_pairs)
    silver_scores2 = cross_encoder2.predict(all_pairs)
    silver_scores = np.array([silver_scores1,silver_scores2]).mean(0)
    silver_samples = list(InputExample(texts=[data[0], data[1]], label=score) for \
        data, score in zip(all_pairs, silver_scores))
    
    del cross_encoder1, cross_encoder2
    torch.cuda.empty_cache()

    ###### Bi-encoder learning ######

    logging.info(f"Loading bi-encoder1 model: {model_name1}")
    logging.info(f"Loading bi-encoder2 model: {model_name2}")
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model1 = models.Transformer(model_name1, max_seq_length=max_seq_length)
    word_embedding_model2 = models.Transformer(model_name2, max_seq_length=max_seq_length)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model1 = models.Pooling(word_embedding_model1.get_word_embedding_dimension(), 
        pooling_mode=args.bi_encoder1_pooling_mode) # bert
    pooling_model2 = models.Pooling(word_embedding_model1.get_word_embedding_dimension(), 
        pooling_mode=args.bi_encoder2_pooling_mode) # roberta

    bi_encoder1 = SentenceTransformer(modules=[word_embedding_model1, pooling_model1], device=device1)
    bi_encoder2 = SentenceTransformer(modules=[word_embedding_model2, pooling_model2], device=device2)

    train_dataloader = DataLoader(silver_samples, shuffle=True, batch_size=batch_size_bi_encoder)
    train_loss1 = losses.CosineSimilarityLoss(model=bi_encoder1)
    train_loss2 = losses.CosineSimilarityLoss(model=bi_encoder2)

    if args.task in ["sts", "sickr", "sts_sickr", "custom"]:
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name="dev")
    else:
        evaluator = EmbeddingSimilarityEvaluatorAUC.from_input_examples(dev_samples, name="dev")

    # Configure the training.
    #warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
    #logging.info(f"Warmup-steps: {warmup_steps}")

    bi_encoder_path1 = f"output/bi-encoder/" \
        f"{args.task}_cycle{cycle}_mutual_parallel_{model_name1.replace('/', '-')}-{start_time}" 
    bi_encoder_path2 = f"output/bi-encoder/" \
        f"{args.task}_cycle{cycle}_mutual_parallel_{model_name2.replace('/', '-')}-{start_time}" 

    bi_encoder1.fit(
        train_objectives=[(train_dataloader, train_loss1)],
        evaluator=evaluator,
        epochs=num_epochs_bi_encoder,
        evaluation_steps=200,
        warmup_steps=0,
        output_path=bi_encoder_path1,
        optimizer_params= {"lr": 5e-5},
        use_amp=True,
    )
    
    bi_encoder2.fit(
        train_objectives=[(train_dataloader, train_loss2)],
        evaluator=evaluator,
        epochs=num_epochs_bi_encoder,
        evaluation_steps=200,
        warmup_steps=0,
        output_path=bi_encoder_path2,
        optimizer_params= {"lr": 5e-5},
        use_amp=True,
    )

    bi_encoder1 = SentenceTransformer(bi_encoder_path1, device=device1)
    bi_encoder2 = SentenceTransformer(bi_encoder_path2, device=device2)

    dev_score1 = evaluator(bi_encoder1)
    dev_score2 = evaluator(bi_encoder2)
    bi_encoder_dev_scores.append([dev_score1, dev_score2])
    logging.info (f"***** dev's spearman's rho: bi-encoder1 {dev_score1:.4f}, bi-encoder2 {dev_score2:.4f} *****")

del bi_encoder1, bi_encoder2
torch.cuda.empty_cache()

print (cross_encoder_dev_scores)
print (bi_encoder_dev_scores)

# best bi-encoder
logging.info ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
bi_encoder_dev_scores1 = [p[0] for p in bi_encoder_dev_scores]
bi_encoder_dev_scores2 = [p[1] for p in bi_encoder_dev_scores]
best_cycle_bi_encoder1 = np.argmax(bi_encoder_dev_scores1)+1
best_cycle_bi_encoder2 = np.argmax(bi_encoder_dev_scores2)+1

best_cycle_bi_encoder_path1 = f"output/bi-encoder/" \
    f"{args.task}_cycle{best_cycle_bi_encoder1}_mutual_parallel_{model_name1.replace('/', '-')}-{start_time}" 
best_cycle_bi_encoder_path2 = f"output/bi-encoder/" \
    f"{args.task}_cycle{best_cycle_bi_encoder2}_mutual_parallel_{model_name2.replace('/', '-')}-{start_time}"

# eval bi-encoder
logging.info ("£££££ Evaluate best bi-encoder1...")
bi_encoder1 = SentenceTransformer(best_cycle_bi_encoder_path1, device=device1)
logging.info (best_cycle_bi_encoder_path1)
eval_encoder(all_test, bi_encoder1, task=args.task, enc_type="bi")

logging.info ("£££££ Evaluate best bi-encoder2...")
bi_encoder2 = SentenceTransformer(best_cycle_bi_encoder_path2, device=device2)
logging.info (best_cycle_bi_encoder_path2)
eval_encoder(all_test, bi_encoder2, task=args.task, enc_type="bi")

# eval bi-encoder (ensembled)
logging.info ("£££££ Evaluate best bi-encoders (ensembled)...")
eval_encoder(all_test, [bi_encoder1, bi_encoder2], task=args.task, enc_type="bi", ensemble=True)

# best cross-encoder
logging.info ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
cross_encoder_dev_scores1 = [p[0] for p in cross_encoder_dev_scores]
cross_encoder_dev_scores2 = [p[1] for p in cross_encoder_dev_scores]
best_cycle_cross_encoder1 = np.argmax(cross_encoder_dev_scores1)+1
best_cycle_cross_encoder2 = np.argmax(cross_encoder_dev_scores2)+1

best_cycle_cross_encoder_path1 = f"output/cross-encoder/" \
    f"{args.task}_cycle{best_cycle_cross_encoder1}_mutual_parallel_{model_name1.replace('/', '-')}-{start_time}" 
best_cycle_cross_encoder_path2 = f"output/cross-encoder/" \
    f"{args.task}_cycle{best_cycle_cross_encoder2}_mutual_parallel_{model_name2.replace('/', '-')}-{start_time}"


# eval cross-encoder
logging.info ("£££££ Evaluate best cross-encoder1...")
logging.info (best_cycle_cross_encoder_path1)
#cross_encoder1 = CrossEncoder(best_cycle_cross_encoder_path1, max_length=64, device=device1)
cross_encoder1 = CrossEncoder(best_cycle_cross_encoder_path1, device=device1)
eval_encoder(all_test, cross_encoder1, task=args.task, enc_type="cross")

logging.info ("£££££ Evaluate best cross-encoder2...")
logging.info (best_cycle_cross_encoder_path2)
#cross_encoder2 = CrossEncoder(best_cycle_cross_encoder_path2, max_length=64, device=device2)
cross_encoder2 = CrossEncoder(best_cycle_cross_encoder_path2, device=device2)
eval_encoder(all_test, cross_encoder2, task=args.task, enc_type="cross")

# eval cross-encoder (ensembled)
logging.info ("£££££ Evaluate best cross-encoders (ensembled)...")
eval_encoder(all_test, [cross_encoder1, cross_encoder2], task=args.task, enc_type="cross", ensemble=True)


logging.info ("\n")
print (args)
logging.info ("\n")
logging.info ("***** END *****")
