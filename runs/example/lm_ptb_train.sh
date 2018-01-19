ROOT=/home/jwei/seq2seq-parse

python $ROOT/main.py \
    --run runs/example/ \
    --model LSTMLanguageModel \
    --train data/lm/train \
    --dev data/lm/valid \
    --in_vocab data/lm/vocab \
    --out_vocab data/lm/vocab \
    --format lm \
    --val_metric perplexity \
    --cutoff 0 \
    --mem 11264 \
    --gpus 1 \
    --imports lm \
    --checkpoint lm.model \
    --epochs 300 \
    --trainer sgd \
    --lr 1 \
    --lr_decay 0.93 \
    --patience 3 \
    --monitor none \
    --batch_size 16 \
    --val_batch_size 64
