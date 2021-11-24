# Trans-Encoder: Unsupervised sentence-pair modelling through self- and mutual-distillations

Code repo for paper [**Trans-Encoder: Unsupervised sentence-pair modelling through self- and mutual-distillations
**](https://arxiv.org/pdf/2109.13059.pdf).

## Dependencies

```
torch=1.8.1
transformers=4.9.0
sentence-transformers=2.0.0
```
Please view `requirements.txt' for more details.

## Train

Self-distillation:
```bash
>> bash train_self_distill.sh 0
```
`0` denotes GPU device index.

Mutual-distillation (two GPUs needed):
```bash
>> bash train_mutual_distill.sh 1,2
```

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

```bash
>> python src/eval.py
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

