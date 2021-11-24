# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from typing import List
import os
import csv
from sentence_transformers.readers import InputExample

from ..utils import write_csv_log

logger = logging.getLogger(__name__)

class CECorrelationEvaluatorAUC:
    """
    This evaluator can be used with the CrossEncoder class. Given sentence pairs and continuous scores,
    it compute the pearson & spearman correlation between the predicted score for the sentence pair
    and the gold score.
    """
    def __init__(self, sentence_pairs: List[List[str]], scores: List[float], name: str='', write_csv: bool = True):
        self.sentence_pairs = sentence_pairs
        self.scores = scores
        self.name = name

        self.csv_file = self.__class__.__name__ + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "roc_auc_score"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        scores = []

        for example in examples:
            sentence_pairs.append(example.texts)
            scores.append(example.label)
        return cls(sentence_pairs, scores, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info(self.__class__.__name__+": Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)


        eval_auc = roc_auc_score(self.scores, pred_scores)

        logger.info("roc_auc_score: {:.4f}".format(eval_auc))

        if output_path is not None and self.write_csv:
            things_to_write = [epoch, steps, eval_auc]
            write_csv_log(output_path=output_path, csv_file=self.csv_file, csv_headers=self.csv_headers, things_to_write=things_to_write)

        return eval_auc
