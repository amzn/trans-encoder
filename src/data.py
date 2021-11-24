# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset
from sentence_transformers.readers import InputExample
from sentence_transformers import util
import pandas as pd
import os
import gzip
import csv
import logging
import subprocess
from zipfile import ZipFile

SCORE = "score"
SPLIT = "split"
SENTENCE = "sentence"
SENTENCE1 = "sentence1"
SENTENCE2 = "sentence2"
QUESTION = "question"
QUESTION1 = "question1"
QUESTION2 = "question2"

def load_snli():
    """
    Load the SNLI dataset (https://nlp.stanford.edu/projects/snli/) from huggingface dataset portal.
    Parameters
    ----------
        None
    Returns
    ----------
        all_pairs: a list of sentence pairs from the SNLI dataset
    """

    all_pairs = []

    dataset = load_dataset("snli")
    all_pairs += [(row["premise"], row["hypothesis"]) for row in dataset["train"]]
    all_pairs += [(row["premise"], row["hypothesis"]) for row in dataset["validation"]]
    all_pairs += [(row["premise"], row["hypothesis"]) for row in dataset["test"]]

    return all_pairs, None, None

def load_sts():
    """
    Load the STS datasets:
        STS 2012: https://www.cs.york.ac.uk/semeval-2012/task6/
        STS 2013: http://ixa2.si.ehu.eus/sts/
        STS 2014: https://alt.qcri.org/semeval2014/task10/
        STS 2015: https://alt.qcri.org/semeval2015/task2/
        STS 2016: https://alt.qcri.org/semeval2016/task1/
        STS-Benchmark: http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark
    Parameters
    ----------
        None
    Returns
    ----------
        all_pairs: a list of sentence pairs from the STS datasets
        all_test: a dict of all test sets of the STS datasets
        dev_samples: a list of InputExample instances as the dev set
    """

    # Check if STS datasets exsist. If not, download and extract it
    sts_dataset_path = "data/"
    if not os.path.exists(os.path.join(sts_dataset_path, "STS_data")):
        logging.info("Dataset not found. Download")
        zip_save_path = "data/STS_data.zip"
        #os.system("wget https://fangyuliu.me/data/STS_data.zip  -P data/")
        subprocess.run(["wget", "--no-check-certificate", "https://fangyuliu.me/data/STS_data.zip", "-P", "data/"])
        with ZipFile(zip_save_path, "r") as zipIn:
            zipIn.extractall(sts_dataset_path)

    all_pairs = []
    all_test = {}
    dedup = set()

    # read sts 2012-2016
    for year in ["2012","2013","2014","2015","2016"]:
        all_test[year] = []
    for year in ["2012","2013","2014","2015","2016"]:
        df = pd.read_csv(f"data/STS_data/en/{year}.test.tsv", delimiter="\t",
            quoting=csv.QUOTE_NONE, encoding="utf-8", names=[SCORE, SENTENCE1, SENTENCE2])
        for row in df.iterrows():
            if str(row[1][SCORE]) == "nan": continue
            all_test[year].append(InputExample(texts=[row[1][SENTENCE1], row[1][SENTENCE2]], label=row[1][SCORE]))

    df = pd.read_csv("data/STS_data/en/2012_to_2016.test.tsv", delimiter="\t",
        quoting=csv.QUOTE_NONE, encoding="utf-8", names=[SCORE, SENTENCE1, SENTENCE2])

    for row in df.iterrows():
        concat = row[1][SENTENCE1]+row[1][SENTENCE2]
        if concat in dedup:
            continue
        all_pairs.append([row[1][SENTENCE1], row[1][SENTENCE2]])
        dedup.add(concat)

    # sts-b
    # Check if STS-B exsists. If not, download and extract it
    sts_dataset_path = "data/stsbenchmark.tsv.gz"
    if not os.path.exists(sts_dataset_path):
        util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)
    # read sts-b
    dev_samples_stsb = []
    test_samples_stsb = []
    with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row[SCORE]) / 5.0  # Normalize score to range 0 ... 1

            if row[SPLIT] == "dev":
                dev_samples_stsb.append(InputExample(texts=[row[SENTENCE1], row[SENTENCE2]], label=score))
            elif row[SPLIT] == "test":
                test_samples_stsb.append(InputExample(texts=[row[SENTENCE1], row[SENTENCE2]], label=score))

            # add (non-duplicated) sentence pair to all_pairs 
            concat = row[SENTENCE1]+row[SENTENCE2]
            if concat in dedup:
                continue
            all_pairs.append([row[SENTENCE1], row[SENTENCE2]])
            dedup.add(concat)

    all_test["stsb"] = test_samples_stsb
    dev_samples = dev_samples_stsb
    return all_pairs, all_test, dev_samples

def load_sickr():
    """
    Load the SICK-R dataset: http://clic.cimec.unitn.it/composes/sick.html
    Parameters
    ----------
        None
    Returns
    ----------
        all_pairs: a list of sentence pairs from the SICK-R dataset
        all_test: a dict of all test sets 
        dev_samples: a list of InputExample instances as the dev set
    """
    

    sts_dataset_path = "data/"
    if not os.path.exists(os.path.join(sts_dataset_path, "STS_data")):
        logging.info("Dataset not found. Download")
        zip_save_path = "data/STS_data.zip"
        subprocess.run(["wget", "--no-check-certificate", "https://fangyuliu.me/data/STS_data.zip", "-P", "data/"])
        with ZipFile(zip_save_path, "r") as zipIn:
            zipIn.extractall(sts_dataset_path)

    all_pairs = []
    all_test = {}
    dedup = set()

    # read sickr
    test_samples_sickr = []
    dev_samples_sickr = []

    df = pd.read_csv("data/STS_data/en/SICK_annotated.txt", delimiter="\t",
            quoting=csv.QUOTE_NONE, encoding="utf-8")

    for row in df.iterrows():
        row = row[1]
        score = row["relatedness_score"] / 5.0
        if row["SemEval_set"] == "TEST":
            test_samples_sickr.append(InputExample(texts=[row["sentence_A"], row["sentence_B"]], label=score))
        elif row["SemEval_set"] == "TRIAL":
            dev_samples_sickr.append(InputExample(texts=[row["sentence_A"], row["sentence_B"]], label=score))
        
        concat = row["sentence_A"]+row["sentence_B"]
        if concat in dedup:
            continue
        all_pairs.append([row["sentence_A"], row["sentence_B"]])
        dedup.add(concat)
    
    all_test["sickr"] = test_samples_sickr
    dev_samples = dev_samples_sickr
    return all_pairs, all_test, dev_samples

def load_qqp():
    """
    Load the QQP dataset (https://www.kaggle.com/c/quora-question-pairs) from huggingface dataset portal.
    Parameters
    ----------
        None
    Returns
    ----------
        all_pairs: a list of sentence pairs from the QQP dataset
        all_test: a dict of all test sets 
        dev_samples: a list of InputExample instances as the dev set
    """

    all_pairs = []
    all_test = {}

    dev_samples_qqp = []
    test_samples_qqp = []

    # Check if the QQP dataset exists. If not, download and extract
    qqp_dataset_path = "data/quora-IR-dataset"
    if not os.path.exists(qqp_dataset_path):
        logging.info("Dataset not found. Download")
        zip_save_path = 'data/quora-IR-dataset.zip'
        util.http_get(url='https://sbert.net/datasets/quora-IR-dataset.zip', path=zip_save_path)
        with ZipFile(zip_save_path, 'r') as zipIn:
            zipIn.extractall(qqp_dataset_path)
    
    qqp_datapoints_cut_train = 10000
    qqp_datapoints_cut_val = 1000
    qqp_datapoints_cut_test = 10000

    with open(os.path.join(qqp_dataset_path, "classification/train_pairs.tsv"), encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            if i == qqp_datapoints_cut_train: break
            all_pairs.append([row[QUESTION1], row[QUESTION2]])

    with open(os.path.join(qqp_dataset_path, "classification/dev_pairs.tsv"), encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            if i == qqp_datapoints_cut_val: break
            dev_samples_qqp.append(InputExample(texts=[row[QUESTION1], row[QUESTION2]], label=int(row['is_duplicate'])))
            all_pairs.append([row[QUESTION1], row[QUESTION2]])

    with open(os.path.join(qqp_dataset_path, "classification/test_pairs.tsv"), encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            if i == qqp_datapoints_cut_test: break
            test_samples_qqp.append(InputExample(texts=[row[QUESTION1], row[QUESTION2]], label=int(row['is_duplicate'])))
            all_pairs.append([row[QUESTION1], row[QUESTION2]])

    all_test["qqp"] = test_samples_qqp
    dev_samples = dev_samples_qqp

    return all_pairs, all_test, dev_samples

def load_qnli():   
    """
    Load the QNLI dataset (part of GLUE: https://gluebenchmark.com/) from huggingface dataset portal.
    Parameters
    ----------
        None
    Returns
    ----------
        all_pairs: a list of sentence pairs from the QNLI dataset
        all_test: a dict of all test sets 
        dev_samples: a list of InputExample instances as the dev set
    """

    all_pairs = []
    all_test = {}

    dev_samples_qnli = []
    test_samples_qnli = []

    dataset = load_dataset("glue", "qnli")

    qnli_datapoints_cut_train = 10000

    for i, row in enumerate(dataset["train"]):
        if i == qnli_datapoints_cut_train: break
        all_pairs.append([row[QUESTION], row[SENTENCE]])

    for row in dataset["validation"]:
        label = 0 if row["label"]==1 else 1
        dev_samples_qnli.append(
            InputExample(texts=[row[QUESTION], row[SENTENCE]], label=label))
        all_pairs.append([row[QUESTION], row[SENTENCE]])
    
    # test labels of qnli are not given, use the first 1k in dev set as test
    all_test["qnli"] = dev_samples_qnli[1000:]
    dev_samples = dev_samples_qnli[:1000]

    return all_pairs, all_test, dev_samples

def load_mrpc():
    """
    Load the MRPC dataset (https://www.microsoft.com/en-us/download/details.aspx?id=52398) from huggingface dataset portal.
    Parameters
    ----------
        None
    Returns
    ----------
        all_pairs: a list of sentence pairs from the MRPC dataset
        all_test: a dict of all test sets 
        dev_samples: a list of InputExample instances as the dev set
    """
    all_pairs = []
    all_test = {}

    dev_samples_mrpc = []
    test_samples_mrpc = []

    dataset = load_dataset("glue", "mrpc")

    for row in dataset["train"]:
        all_pairs.append([row[SENTENCE1], row[SENTENCE2]])

    for row in dataset["validation"]:
        dev_samples_mrpc.append(
            InputExample(texts=[row[SENTENCE1], row[SENTENCE2]], label=int(row["label"])))
        all_pairs.append([row[SENTENCE1], row[SENTENCE2]])
        
    for row in dataset["test"]:
        test_samples_mrpc.append(
            InputExample(texts=[row[SENTENCE1], row[SENTENCE2]], label=int(row["label"])))
        all_pairs.append([row[SENTENCE1], row[SENTENCE2]])
    
    all_test["mrpc"] = test_samples_mrpc
    dev_samples = dev_samples_mrpc
    
    return all_pairs, all_test, dev_samples

def load_sts_and_sickr():
    """
    Load both STS and SICK-R datasets. Use STS-B's dev set for dev.
    Parameters
    ----------
        None
    Returns
    ----------
        all_pairs: a list of sentence pairs from the STS+SICK-R dataset
        all_test: a dict of all test sets 
        dev_samples: a list of InputExample instances as the dev set
    """

    all_pairs_sts, all_test_sts, dev_samples_sts = load_sts()
    all_pairs_sickr, all_test_sickr, dev_samples_sickr = load_sickr()
    all_pairs = all_pairs_sts+all_pairs_sickr
    all_test = {**all_test_sts, **all_test_sickr}
    return all_pairs, all_test, dev_samples_sts # sts-b's dev is used

def load_custom(fpath):
    """
    Load custom sentence-pair corpus. Use STS-B's dev set for dev.
    Parameters
    ----------
        fpath: path to the training file, where sentence pairs are formatted as 'sent1||sent2'
    Returns
    ----------
        all_pairs: a list of sentence pairs from the STS+SICK-R dataset
        all_test: a dict of all test sets 
        dev_samples: a list of InputExample instances as the dev set
    """
    all_pairs = []
    all_test = {}

    # load custom training corpus
    with open(fpath, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line.split("||")) != 2: continue # skip
        sent1, sent2 = line.split("||")
        all_pairs.append([sent1, sent2])
    
    # load STS-b dev/test set
    # Check if STS-B exsists. If not, download and extract it
    sts_dataset_path = "data/stsbenchmark.tsv.gz"
    if not os.path.exists(sts_dataset_path):
        util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)
    # read sts-b
    dev_samples_stsb = []
    test_samples_stsb = []
    with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row[SCORE]) / 5.0  # Normalize score to range 0 ... 1

            if row[SPLIT] == "dev":
                dev_samples_stsb.append(InputExample(texts=[row[SENTENCE1], row[SENTENCE2]], label=score))
            elif row[SPLIT] == "test":
                test_samples_stsb.append(InputExample(texts=[row[SENTENCE1], row[SENTENCE2]], label=score))

            # add entence pair to all_pairs 
            #all_pairs.append([row[SENTENCE1], row[SENTENCE2]])

    all_test["stsb"] = test_samples_stsb
    dev_samples = dev_samples_stsb

    return all_pairs, all_test, dev_samples


task_loader_dict = {
    "sts": load_sts, 
    "sickr": load_sickr,
    "sts_sickr": load_sts_and_sickr,
    "qqp": load_qqp,
    "qnli": load_qnli,
    "mrpc": load_mrpc,
    "snli": load_snli,
    "custom": load_custom
}

def load_data(task, fpath=None):
    """
    A unified dataset loader for all tasks.
    Parameters
    ----------
        task: a string specifying dataset/task to be loaded (for possible options see 'task_loader_dict')
    Returns
    ----------
        all_pairs: a list of sentence pairs from the specified dataset
        all_test: a dict of all test sets 
        dev_samples: a list of InputExample instances as the dev set
    """
    if task not in task_loader_dict.keys():
        raise NotImplementedError()
    if task == "custom":
        return task_loader_dict[task](fpath)
    else:
        return task_loader_dict[task]()


if __name__ == "__main__":
    # test if all datasets can be properly loaded
    for task in task_loader_dict:
        print (f"loading {task}...")
        load_data(task)
        print ("done.")