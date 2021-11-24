# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from typing import List
import numpy as np
import os
import csv
from sentence_transformers.readers import InputExample

from .CECorrelationEvaluatorEnsemble import CECorrelationEvaluatorEnsemble
from ..utils import write_csv_log

logger = logging.getLogger(__name__)

class CECorrelationEvaluatorAUCEnsemble(CECorrelationEvaluatorEnsemble):
    """
    This evaluator can be used with the CrossEncoder class. Given sentence pairs and continuous scores,
    it compute the pearson & spearman correlation between the predicted score for the sentence pair
    and the gold score.
    """
    def __init__(self, sentence_pairs: List[List[str]], scores: List[float], name: str='', write_csv: bool = True):
        CECorrelationEvaluatorEnsemble.__init__(self, sentence_pairs, scores, name, write_csv)
        self.csv_headers = ["epoch", "steps", "roc_auc_score"] # overwrite parent's csv_headers

    def __call__(self, models, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info(self.__class__.__name__+": Evaluating the model on " + self.name + " dataset" + out_txt)

        all_scores = []
        for model in models:
            pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
            all_scores.append(pred_scores)
            
        pred_scores = np.array(all_scores).mean(0)

        eval_auc = roc_auc_score(self.scores, pred_scores)

        logger.info("roc_auc_score: {:.4f}".format(eval_auc))
        
        if output_path is not None and self.write_csv:
            things_to_write = [epoch, steps, eval_auc]
            write_csv_log(output_path=output_path, csv_file=self.csv_file, csv_headers=self.csv_headers, things_to_write=things_to_write)

        return eval_auc
