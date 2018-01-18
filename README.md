# A Dynet Framework (Python)

A Dynet Framework written by Johnny Wei (jwei@umass.edu). The goal is to make it easy to experiment with new models. The code should provide basic handling of input/output, training, testing, and validation. A few baselines for language and sequence to sequence modelling are provided. Don't be afraid to look into the source - take a look at the baselines as examples! If it's not obvious on how to extend this for your needs, let me know through github or email. 

The framework is still largely a work in progress. I'd like to add more examples (e.g. classification, sequence tagging etc.) to the framework.

## Training an LSTM Language Model

To train an LSTM language model with the specified parameters in `lm.py`, first specify the framework directory and run the example script with:
  export SEQ2SEQROOT=/directory/of/repository
  bash runs/example/experiment.sh
The memory specified in the example script is tuned for a Titan X. With a Titan X I was able to train at 158s per epoch, and after epoch 7 I had:
  Epoch 7. Time elapsed: 158s, 42068/42068. Total Loss: 4693762.2395. Average sequence loss: 111.5756. Average Token Loss: 5.0493.
  Done. Total loss: 4693762.239487
  [lr=0.64699 clips=2630 updates=2630]
  Validating...
  Done. Validation loss: 363955.496216. Average Token loss: 4.934321. Perplexity: 138.978705.
