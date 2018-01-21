# A Dynet Framework (Python)

A Dynet framework written by Johnny Wei (jwei@umass.edu). The goal is to make it easy to experiment with new models by providing basic handling of input/output, training, testing, and validation. A few baselines for language and sequence to sequence modelling are also provided. Don't be afraid to look into the source - take a look at the baselines as examples! If it's not obvious on how to extend this for your needs, let me know through github or email. 

The framework is still largely a work in progress. I'd like to add more examples (e.g. classification, sequence tagging etc.) to the framework. 
<br>

## Training an LSTM Language Model

To train an LSTM language model with the specified hyperparameters in `lm.py`, first specify the framework directory and run the example script with:

    export SEQ2SEQROOT=/directory/of/repository
    bash runs/example/lm_ptb_train.py

The hyperparmeters specify a very large language model (512 input dim, 1024 hidden dim, 3 layer LSTM)! I trained using a Titan X (12GB), and tuned the memory accordingly. With it, training was possible at 383s per epoch, and after epoch 31 the validation shows:

    Epoch 31. Time elapsed: 383s, 42068/42068. Total Loss: 3759063.9351. Average sequence loss: 89.3568. Average Token Loss: 4.0438.
    Done. Total loss: 3759063.935143
    [lr=0.00763076 clips=2630 updates=2630]
    Validating...
    Done. Validation loss: 338222.015259. Average Token loss: 4.585439. Perplexity: 98.046266.
    Monitored quantity improved.
    
The validation results and checkpointed models should be saved in `runs/example/` (specified in the script). Test the checkpointed model with:

    bash runs/example/lm_ptb_test.sh

I was able to achieve these results on the test set:
    
    Testing model on metric (perplexity).
    Testing...
    Done. Validation loss: 375382.022034. Average Token loss: 4.553949. Perplexity: 95.006857.

For reference, [Zaremba et al., 2014](https://arxiv.org/abs/1409.2329) achieves 82.2 perplexity on validation, and 78.4 on test with a 2 layer, 1500 hidden dim LSTM. This difference in performance needs to be investigated.
