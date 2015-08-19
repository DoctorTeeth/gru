import numpy as np

class Adagrad(object):

  def __init__(self, weights, learning_rate):
    """
    weights is a list of tensors
    we will need to generate a copy of these in order to do
    adagrad updates
    """
    self.lr = learning_rate
    
    # Initialize adagrad memory tensors
    self.mems = []
    for tensor in weights:
      self.mems.append(np.zeros_like(tensor))

  def update_weights(self, params, dparams):
    for param, dparam, mem in zip(params, dparams, self.mems):
      dparam = np.clip(dparam,-1,1) # clip so gradients don't explode
      mem += dparam*dparam
      param += -self.lr * dparam / np.sqrt(mem + 1e-8) # adagrad update
