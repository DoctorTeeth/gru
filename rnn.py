"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

class RNN(object):

  def __init__(self, in_size, out_size, hidden_size):
    
    # the model parameters
    self.Wxh = np.random.randn(hidden_size, in_size)*0.01 # input to hidden
    self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    self.Why = np.random.randn(out_size, hidden_size)*0.01 # hidden to output
    self.bh  = np.zeros((hidden_size, 1)) # hidden bias
    self.by  = np.zeros((out_size, 1)) # output bias

    self.weights = [self.Wxh,self.Whh,self.Why,self.bh,self.by]
    self.names = ["Wxh","Whh","Why","bh","by"]

    # the grads w.r.t the model parameters

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
    #ps are the activation of last layer - probability

    # so hs is actually a dict
    # this affects computation of our first hidden state
    # we reset hidden state after every sequence
    # TODO: maybe we don't need to 
    hs[-1] = np.zeros((self.Wxh.shape[0],1))
    #print hs
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

    # go backwards through time
    for t in reversed(xrange(len(inputs))):

      # we have dC/dy_i = p_i - answer_i
      # by the deriv of NLL wrt softmax
      dy = np.copy(ps[t])
      dy -= targets[t].T # backprop into y

      # for weights from hidden to y
      # y is sum of affine transform and a bias
      # backprop shares grad between sum elements
      # so grad wrt an individual weight w_hy
      # when y = w_hy*h + ... + bias
      # in this case, through the chain rule
      # we come out to dy times the hidden unit 
      # activation that the weight is associated with
      dWhy += np.dot(dy, hs[t].T)

      # for y bias
      # y is the sum of an affine transform and a bias
      # in backprop, grads get copied over sums, so the grad
      # wrt the whole affine transform is just dy, and so is the
      # grad with respect to the bias
      # so dC/dby = dC/dy dy/dby = dy
      dby += dy

      #TODO: give a better explanation of this and below
      dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
      h2 = np.multiply(hs[t], hs[t])
      dhraw = np.multiply((1 - h2) , dh) # backprop through tanh nonlinearity
      
      # for the bias into the hidden layer
      dbh += dhraw
      
      # for weights from input to hidden
      dWxh += np.dot(dhraw, xs[t].T)

      # for weights from hidden to hidden
      dWhh += np.dot(dhraw, hs[t-1].T)

      # TODO: what is this?
      dhnext = np.dot(self.Whh.T, dhraw)

    deltas = [dWxh, dWhh, dWhy, dbh, dby]

    return loss, deltas, ps

