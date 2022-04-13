# Trans-Encoder

<img align="right" width="500"  src="https://production-media.paperswithcode.com/methods/e6c08315-2b70-4125-aeb2-147a6785d9b1.png">

Code repo for ICLR 2022 paper **_[Trans-Encoder: Unsupervised sentence-pair modelling through self- and mutual-distillations](https://arxiv.org/abs/2109.13059)_** <br>
by [Fangyu Liu](http://fangyuliu.me/about.html), [Yunlong Jiao](https://yunlongjiao.github.io/), [Jordan Massiah](https://www.linkedin.com/in/jordan-massiah-562862136/?originalSubdomain=uk), [Emine Yilmaz](https://sites.google.com/site/emineyilmaz/), [Serhii Havrylov](https://serhii-havrylov.github.io/).

Trans-Encoder is a state-of-the-art unsupervised sentence similarity model. It conducts self-knowledge distillation on top of pretrained language models by alternating between their bi- and cross-encoder forms.


## Huggingface pretrained models for STS

<table>
<tr><th> base models </th><th> large models </th></tr>
<tr><td>

|model | STS avg. |
|--------|--------|
|baseline: [unsup-simcse-bert-base](https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased) | 76.21 |
| [trans-encoder-bi-simcse-bert-base](https://huggingface.co/cambridgeltl/trans-encoder-bi-simcse-bert-base) | 80.41  |
| [trans-encoder-cross-simcse-bert-base](https://huggingface.co/cambridgeltl/trans-encoder-cross-simcse-bert-base) | 79.90  |
|baseline:  [unsup-simcse-roberta-base](https://huggingface.co/princeton-nlp/unsup-simcse-roberta-base) | 76.10 |
| [trans-encoder-bi-simcse-roberta-base](https://huggingface.co/cambridgeltl/trans-encoder-bi-simcse-roberta-base) | 80.47 |
| [trans-encoder-cross-simcse-roberta-base](https://huggingface.co/cambridgeltl/trans-encoder-cross-simcse-roberta-base) | **81.15** |
</td><td>

|model | STS avg. |
|--------|--------|
|baseline:  [unsup-simcse-bert-large](https://huggingface.co/princeton-nlp/unsup-simcse-bert-large-uncased) | 78.42 |
| [trans-encoder-bi-simcse-bert-large](https://huggingface.co/cambridgeltl/trans-encoder-bi-simcse-bert-large) |  82.65  |
| [trans-encoder-cross-simcse-bert-large](https://huggingface.co/cambridgeltl/trans-encoder-cross-simcse-bert-large) |  82.52 |
|baseline:  [unsup-simcse-roberta-large](https://huggingface.co/princeton-nlp/unsup-simcse-roberta-large) | 78.92 |
| [trans-encoder-bi-simcse-roberta-large](https://huggingface.co/cambridgeltl/trans-encoder-bi-simcse-roberta-large) | **82.93** |
| [trans-encoder-cross-simcse-roberta-large](https://huggingface.co/cambridgeltl/trans-encoder-cross-simcse-roberta-large) |  **82.93** |


</td></tr> </table>


## Dependencies

```
torch==1.8.1
transformers==4.9.0
sentence-transformers==2.0.0
```
Please view [requirements.txt](https://github.com/amzn/trans-encoder/blob/main/requirements.txt) for more details.

## Data
All training and evaluation data will be automatically downloaded when running the scripts. See `data.py` for details.

## Train

Self-distillation:
```bash
>> bash train_self_distill.sh 0
```
`0` denotes GPU device index.

Mutual-distillation (two GPUs needed; by default using SimCSE-BERT and RoBERTa for ensembling):
```bash
>> bash train_mutual_distill.sh 0,1
```

`--task` options: `sts` (STS2012-2016 and STS-b), `sickr`, `sts_sickr` (STS2012-2016, STS-b, and SICK-R), `qqp`, `qnli`, `mrpc`, `snli`, `custom`. See [src/data.py](https://github.com/amzn/trans-encoder/blob/main/src/data.py) for task data details.
`--use_large` (`store_true`): availible in [src/train_mutual_distill.py](https://github.com/amzn/trans-encoder/blob/main/src/mutual_distill_parallel.py), switch to large models instead of base models.

Train with your custom corpus:
```bash
>> CUDA_VISIBLE_DEVICES=0,1 python src/mutual_distill_parallel.py \
         --batch_size_bi_encoder 128 \
         --batch_size_cross_encoder 64 \
         --num_epochs_bi_encoder 10 \
         --num_epochs_cross_encoder 1 \
         --cycle 3 \
         --bi_encoder1_pooling_mode cls \
         --bi_encoder2_pooling_mode cls \
         --init_with_new_models \
         --task custom \
         --random_seed 2021 \
         --custom_corpus_path CORPUS_PATH
```
`CORPUS_PATH` should point to your custom corpus in which every line should be a sentence pair in the form of `sent1||sent2`.

## Evaluate
### Evaluate a single model

Bi-encoder:
```bash
>> python src/eval.py \
--model_name_or_path "cambridgeltl/trans-encoder-bi-simcse-roberta-large"  \
--mode bi \
--task sts_sickr
```
Cross-encoder:
```bash
>> python src/eval.py \
--model_name_or_path "cambridgeltl/trans-encoder-cross-simcse-roberta-large"  \
--mode cross \
--task sts_sickr
```
### Evaluate an ensembled model

Bi-encoder:
```bash
>> python src/eval.py \
--model_name_or_path1 "cambridgeltl/trans-encoder-bi-simcse-bert-large"  \
--model_name_or_path2 "cambridgeltl/trans-encoder-bi-simcse-roberta-large"  \
--mode bi \
--ensemble \
--task sts_sickr
```

Cross-encoder:
```bash
>> python src/eval.py \
--model_name_or_path1 "cambridgeltl/trans-encoder-cross-simcse-bert-large"  \
--model_name_or_path2 "cambridgeltl/trans-encoder-cross-simcse-roberta-large"  \
--mode cross \
--ensemble \
--task sts_sickr
```

## Authors

- [**Fangyu Liu**](http://fangyuliu.me/about.html): Main contributor

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

