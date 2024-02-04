## Uasge

# BioBERT-PyTorch
This repository provides the PyTorch implementation of [BioBERT](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506).
You can easily use BioBERT with [transformers](https://github.com/huggingface/transformers).
This project is supported by the members of [DMIS-Lab](https://dmis.korea.ac.kr/) @ Korea University including Jinhyuk Lee, Wonjin Yoon, Minbyul Jeong, Mujeen Sung, and Gangwoo Kim.
1.Download the BioBERT v1.1 (+ PubMed 1M) model (or any other model) from the bioBERT repo:https://github.com/naver/biobert-pretrained
2.Extract the downloaded file, e.g. with tar -xzf biobert_v1.1_pubmed.tar.gz
3. Convert the bioBERT model TensorFlow checkpoint to a PyTorch and PyTorch-Transformers compatible one: pytorch_transformers bert biobert_v1.1_pubmed/model.ckpt-1000000 biobert_v1.1_pubmed/bert_config.json biobert_v1.1_pubmed/pytorch_model.bin
4.Move config mv biobert_v1.1_pubmed/bert_config.json biobert_v1.1_pubmed/config.json
5.Then pass the folder name to the --model_name_or_path argument. You can run this simple script to check, if everything works:

## Models
We provide following versions of BioBERT in PyTorch (click [here](https://huggingface.co/dmis-lab) to see all).
You can use BioBERT in `transformers` by setting `--model_name_or_path` as one of them (see example below).
* `dmis-lab/biobert-base-cased-v1.2`: Trained in the same way as BioBERT-Base v1.1 but includes LM head, which can be useful for probing
* `dmis-lab/biobert-base-cased-v1.1`: BioBERT-Base v1.1 (+ PubMed 1M)
* `dmis-lab/biobert-large-cased-v1.1`: BioBERT-Large v1.1 (+ PubMed 1M)
* `dmis-lab/biobert-base-cased-v1.1-mnli`: BioBERT-Base v1.1 pre-trained on MNLI
* `dmis-lab/biobert-base-cased-v1.1-squad`: BioBERT-Base v1.1 pre-trained on SQuAD
* `dmis-lab/biobert-base-cased-v1.2`: BioBERT-Base v1.2 (+ PubMed 1M + LM head)

For instance, to train BioBERT on the NER dataset (NCBI-disease), run as:

# Choose dataset and run
export DATA_DIR=../data
export ENTITY=NCBI-disease
python run_ner.py \
    --data_dir ${DATA_DIR}/${ENTITY} \
    --labels ${DATA_DIR}/${ENTITY}/labels.txt \
    --model_name_or_path modules/biobert-base-cased-v1.1 \
    --output_dir output/${ENTITY} \
    --max_seq_length 128 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --save_steps 1000 \
    --seed 1 \
    --do_train \
    --do_eval \
    --do_predict \

TO BE CONTINUED ..
