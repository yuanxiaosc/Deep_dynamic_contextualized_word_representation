# Deep_dynamic_word_representation
TensorFlow code and pre-trained models for DDWR

# Important explanation
1. The method of the model is simple, only using the feed forward neural network with attention mechanism.
2. Model training is fast, and only a few cycles can be used to train the model. The value of the initialization parameter comes from the BERT model of Google.
3. The effect of the model is very good. In most cases, it is consistent with the current (2018-11-13) optimal model. Sometimes the effect is better. The optimal effect can be seen in [gluebenchmark](https://gluebenchmark.com/leaderboard).

# Thought of article

This model Deep_dynamic_word_representation(DDWR) combines the BERT model and ELMo's deep context word representation.

The BERT comes from [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
The ELMo comes from [Deep contextualized word representations](https://arxiv.org/abs/1802.05365v2)

# Basic usage method

## Download Pre-trained models

[BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)

## Doenload [GLUE data](https://gluebenchmark.com/tasks)DATA

using this [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)

## Sentence (and sentence-pair) classification tasks

```
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

python run_classifier_elmo.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/mrpc_output/
```

### Prediction from classifier

```
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue
export TRAINED_CLASSIFIER=/path/to/fine/tuned/classifier

python run_classifier_elmo.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/tmp/mrpc_output/
```
more methods to [google-research/bert](https://github.com/google-research/bert)



