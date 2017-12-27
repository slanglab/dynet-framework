ROOT=/home/johntzwei/Documents/vanilla-seq2seq

python $ROOT/main.py \
    --run runs/example/ \
    --model LSTMLanguageModel \
    --train data/lm/train \
    --dev data/lm/valid \
    --in_vocab data/lm/vocab \
    --out_vocab data/lm/vocab \
    --format lm \
    --val_metric perplexity \
    --cutoff 39800 \
    --mem 256 \
    --gpus 0 \
    --imports lm \
    --checkpoint lm.model \
    --batch_size 16 \
    --val_batch_size 16
