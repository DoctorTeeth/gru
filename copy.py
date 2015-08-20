import numpy as np
import sequences 
from rnn import RNN
from gru import GRU
from optimizers import Adagrad
from util import gradCheck
import sys

# Uncomment to remove determinism
np.random.seed(0)

# Set to True to perform gradient checking
GRAD_CHECK = False

vec_size = 8
out_size = vec_size # Size of output bit vector at each time step
in_size = vec_size + 2 # Input vector size, bigger because of start+stop bits
hidden_size = 100 # Size of hidden layer of neurons
learning_rate = 1e-1

# An object that keeps the network state during training.
model = GRU(in_size, out_size, hidden_size)

# An object that keeps the optimizer state during training
optimizer = Adagrad(model.weights,learning_rate)

n = 0 # counts the number of sequences trained on

while True:

  # train on sequences of length from 1 to 4
  seq_length = np.random.randint(1,5)
  i, t = sequences.copy_sequence(seq_length, vec_size) 
  inputs = np.matrix(i)
  targets = np.matrix(t)

  # forward seq_length characters through the net and fetch gradient
  # deltas is a list of deltas oriented same as list of weights
  loss, deltas, outputs = model.lossFun(inputs, targets)

  # sometimes print out diagnostic info
  if n % 1000 == 0:
    print 'iter %d' % (n) 
    print "inputs: "
    print inputs
    print "outputs: "
    for k in outputs:
      print outputs[k].T

    # calculate the BPC
    print "bpc:"
    # this is actually nats-per-char
    # if we count on the whole sequence it's unfair to 
    print np.sum(loss) / ((seq_length*2 + 2) * vec_size)
  
    if GRAD_CHECK:
      # Check weights using finite differences
      check = gradCheck(model, deltas, inputs, targets, 1e-5, 1e-7)
      print "PASS DIFF CHECK?: ", check
      if not check:
        sys.exit(1)

  optimizer.update_weights(model.weights, deltas)

  n += 1 
