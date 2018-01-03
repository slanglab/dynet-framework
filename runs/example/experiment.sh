ROOT=/home/jwei/seq2seq-parse

python $ROOT/main.py \
    --run runs/example/ \
    --model Seq2SeqVanilla \
    --train data/lm/train \
    --dev data/lm/valid \
    --in_vocab data/lm/vocab \
    --out_vocab data/lm/vocab \
    --format lm \
    --val_metric perplexity \
    --cutoff 0 \
    --mem 22528 \
    --gpus 1 \
    --imports seq2seq \
    --checkpoint lm.model \
    --epochs 600 \
    --trainer sgd \
    --lr 1 \
    --lr_decay 0.5 \
    --monitor train_loss \
    --batch_size 64 \
    --val_batch_size 64
