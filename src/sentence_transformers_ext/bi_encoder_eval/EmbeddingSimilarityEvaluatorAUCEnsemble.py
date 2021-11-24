# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from sentence_transformers.evaluation import (
        SentenceEvaluator,
        SimilarityFunction
)
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score
from sentence_transformers.readers import InputExample

from ..utils import write_csv_log

logger = logging.getLogger(__name__)

class EmbeddingSimilarityEvaluatorAUCEnsemble(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float], batch_size: int = 16, main_similarity: SimilarityFunction = SimilarityFunction.COSINE, name: str = '', show_progress_bar: bool = False, write_csv: bool = True):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.write_csv = write_csv

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = self.__class__.__name__ + ("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_auc", "euclidean_auc", "manhattan_auc", "dot_auc"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)


    def __call__(self, models, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info(self.__class__.__name__+": Evaluating the model on " + self.name + " dataset" + out_txt)

        labels = self.scores

        auc_cosine_scores_all_models = []
        auc_manhattan_distances_all_models = []
        auc_euclidean_distances_all_models = []
        auc_dot_products_all_models = []
        
        # compute average predictions of all models
        for model in models:
            embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
            embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
            
            cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
            manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
            euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
            dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]
            auc_cosine_scores_all_models.append(roc_auc_score(labels, cosine_scores))
            auc_manhattan_distances_all_models.append(roc_auc_score(labels, manhattan_distances))
            auc_euclidean_distances_all_models.append(roc_auc_score(labels, euclidean_distances))
            auc_dot_products_all_models.append(roc_auc_score(labels, dot_products))

        eval_auc_cosine = np.array(auc_cosine_scores_all_models).mean(0)
        eval_auc_manhattan = np.array(auc_manhattan_distances_all_models).mean(0)
        eval_auc_euclidean = np.array(auc_euclidean_distances_all_models).mean(0)
        eval_auc_dot = np.array(auc_dot_products_all_models).mean(0)

        logger.info("Cosine-Similarity AUC: {:.4f}".format(eval_auc_cosine))
        logger.info("Manhattan-Distance AUC: {:.4f}".format(eval_auc_manhattan))
        logger.info("Euclidean-Distance AUC: {:.4f}".format(eval_auc_euclidean))
        logger.info("Dot-Product-Similarity AUC: {:.4f}".format(eval_auc_dot))

        if output_path is not None and self.write_csv:
            things_to_write = [epoch, steps, eval_auc_cosine, eval_auc_euclidean,
                                 eval_auc_manhattan, eval_auc_dot]
            write_csv_log(output_path=output_path, csv_file=self.csv_file, csv_headers=self.csv_headers, things_to_write=things_to_write)

        if self.main_similarity == SimilarityFunction.COSINE:
            return eval_auc_cosine
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return eval_auc_euclidean
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return eval_auc_manhattan
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return eval_spearman_dot
        elif self.main_similarity is None:
            return max(eval_auc_cosine, eval_auc_manhattan, eval_auc_euclidean, eval_auc_dot)
        else:
            raise ValueError("Unknown main_similarity value")
