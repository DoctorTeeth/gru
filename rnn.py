"""
Adapted from:
  Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
  BSD License
"""
import numpy as np

class RNN(object):

  def __init__(self, in_size, out_size, hidden_size):
    

    self.Wxh = np.random.randn(hidden_size, in_size)*0.01 # input to hidden
    self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    self.Why = np.random.randn(out_size, hidden_size)*0.01 # hidden to output
    self.bh  = np.zeros((hidden_size, 1)) # hidden bias
    self.by  = np.zeros((out_size, 1)) # output bias

    self.weights = [self.Wxh,self.Whh,self.Why,self.bh,self.by]
    self.names = ["Wxh","Whh","Why","bh","by"]

  def lossFun(self, inputs, targets):
    """
    inputs,targets are both list of integers.
    where in this case, H is hidden_size from above
    returns the loss, gradients on model parameters, and last hidden state
    n is the counter we're on, just for debugging
    """

    xs, hs, ys, ps = {}, {}, {}, {}
    #xs are inputs
    #hs are hiddens
    #ys are outputs
    #ps are the activation of last layer

    # we reset hidden state after every sequence
    hs[-1] = np.zeros((self.Wxh.shape[0],1))
    loss = 0

    # forward pass, compute outputs, t indexes time
    for t in xrange(len(inputs)):
      
      #xs represents the input vector at this time
      xs[t] = np.matrix(inputs[t]).T

      # hidden layer <- current input, last hidden,  bias
      hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)

      # pre-activation output <- current hidden, bias
      ys[t] = np.dot(self.Why, hs[t]) + self.by

      # the outputs use a sigmoid nonlinearity
      ps[t] = 1 / (1 + np.exp(-ys[t]))

      #create a vector of all ones the size of our outpout
      one = np.ones_like(ps[t])

      # compute the vectorized cross-entropy loss
      a = np.multiply(targets[t].T , np.log(ps[t]))
      b = np.multiply(one - targets[t].T, np.log(one-ps[t]))
      loss -= (a + b) 

    # backward pass: compute gradients going backwards
    # allocate space for the grads of loss with respect to the weights
    dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)

    # allocate space for the grads of loss with respect to biases
    dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
    
    # no error is received from beyond the end of the sequence
    dhnext = np.zeros_like(hs[0])

    # bprop to compute grads
    for t in reversed(xrange(len(inputs))):

      dy = np.copy(ps[t])
      dy -= targets[t].T # backprop into y

      dWhy += np.dot(dy, hs[t].T)

      dby += dy

      # h[t] influences cost through y[t] and h[t+1]
      dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
      h2 = np.multiply(hs[t], hs[t])
      dhraw = np.multiply((1 - h2) , dh) # backprop through tanh nonlinearity
      
      dbh += dhraw
      
      dWxh += np.dot(dhraw, xs[t].T)

      dWhh += np.dot(dhraw, hs[t-1].T)

      # contribution of h[t] to cost through h[t+1] 
      dhnext = np.dot(self.Whh.T, dhraw)

    deltas = [dWxh, dWhh, dWhy, dbh, dby]

    return loss, deltas, ps
