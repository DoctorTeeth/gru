"""
Minimal single layer GRU implementation.
"""
import numpy as np
import sys

class GRU(object):

  def __init__(self, in_size, out_size, hidden_size):
    """
    This class implements a GRU.
    """

    # TODO: go back to 0.01 initialization
    # TODO: use glorot initialization?
    # input weights
    self.Wxc = np.random.randn(hidden_size, in_size)*1 # input to candidate 
    self.Wxr = np.random.randn(hidden_size, in_size)*1 # input to reset
    self.Wxz = np.random.randn(hidden_size, in_size)*1 # input to interpolate

    # Recurrent weights
    self.Rhc = np.random.randn(hidden_size, hidden_size)*1 # hidden to candidate 
    self.Rhr = np.random.randn(hidden_size, hidden_size)*1 # hidden to reset 
    self.Rhz = np.random.randn(hidden_size, hidden_size)*1 # hidden to interpolate 

    # Weights from hidden layer to output layer
    self.Why = np.random.randn(out_size, hidden_size)*1 # hidden to output

    # biases
    self.bc  = np.zeros((hidden_size, 1)) # bias for candidate
    self.br  = np.zeros((hidden_size, 1)) # bias for reset 
    self.bz  = np.zeros((hidden_size, 1)) # bias for interpolate

    self.by  = np.zeros((out_size, 1)) # output bias

    self.weights = [   self.Wxc
                     , self.Wxr
                     , self.Wxz
                     , self.Rhc
                     , self.Rhr
                     , self.Rhz
                     , self.Why
                     , self.bc
                     , self.br
                     , self.bz
                     , self.by
                   ]

    # I used this for grad checking, but I should clean up
    self.names   = [   "Wxc"
                     , "Wxr"
                     , "Wxz"
                     , "Rhc"
                     , "Rhr"
                     , "Rhz"
                     , "Why"
                     , "bc"
                     , "br"
                     , "bz"
                     , "by"
                   ]


  def lossFun(self, inputs, targets):
    """
    Does a forward and backward pass on the network using (inputs, targets)
    inputs is a bit-vector of seq-length
    targets is a bit-vector of seq-length
    """

    xs, rbars, rs, zbars, zs, cbars, cs, hs, ys, ps = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    #xs are inputs
    #hs are hiddens
    #ys are outputs
    #ps are the activation of last layer

    # This resets the hidden state after every new sequence
    # TODO: maybe we don't need to 
    hs[-1] = np.zeros((self.Wxc.shape[0],1))
    loss = 0

    # forward pass, compute outputs, t indexes time
    for t in xrange(len(inputs)):
      
      # For every variable V, Vbar represents the pre-activation version
      # For every variable Q, Qnext represents that variable at time t+1
        # where t is understood from context

      # xs is the input vector at this time
      xs[t] = np.matrix(inputs[t]).T
       
      # The r gate, which modulates how much signal from h[t-1] goes to the candidate
      rbars[t] = np.dot(self.Wxr, xs[t]) + np.dot(self.Rhr, hs[t-1]) + self.br
      rs[t] = 1 / (1 + np.exp(-rbars[t]))
      # TODO: use an already existing sigmoid function

      # The z gate, which interpolates between candidate and h[t-1] to compute h[t]
      zbars[t] = np.dot(self.Wxz, xs[t]) + np.dot(self.Rhz, hs[t-1]) + self.bz
      zs[t] = 1 / (1 + np.exp(-zbars[t]))
      
      # The candidate, which is computed and used as described above.
      cbars[t] = np.dot(self.Wxc, xs[t]) + np.dot(self.Rhc, np.multiply(rs[t] , hs[t-1])) + self.bc
      cs[t] = np.tanh(cbars[t])

      #TODO: get rid of this
      ones = np.ones_like(zs[t])

      # Compute new h by interpolating between candidate and old h
      hs[t] = np.multiply(cs[t],zs[t]) + np.multiply(hs[t-1],ones - zs[t])

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

    # allocate space for the grads of loss with respect to the weights

    dWxc = np.zeros_like(self.Wxc)
    dWxr = np.zeros_like(self.Wxr)
    dWxz = np.zeros_like(self.Wxz)
    dRhc = np.zeros_like(self.Rhc)
    dRhr = np.zeros_like(self.Rhr)
    dRhz = np.zeros_like(self.Rhz)
    dWhy = np.zeros_like(self.Why)

    # allocate space for the grads of loss with respect to biases
    dbc = np.zeros_like(self.bc)
    dbr = np.zeros_like(self.br)
    dbz = np.zeros_like(self.bz)
    dby = np.zeros_like(self.by)
    
    # no error is received from beyond the end of the sequence
    dhnext = np.zeros_like(hs[0])
    drbarnext = np.zeros_like(rbars[0])
    dzbarnext = np.zeros_like(zbars[0])
    dcbarnext = np.zeros_like(cbars[0])
    zs[len(inputs)] = np.zeros_like(zs[0])
    rs[len(inputs)] = np.zeros_like(rs[0])
    
    # go backwards through time
    for t in reversed(xrange(len(inputs))):

      # For every variable X, dX represents dC/dX 
      # For variables that influence C at multiple time steps,
        # such as the weights, the delta is a sum of deltas at multiple
        # time steps

      dy = np.copy(ps[t])
      dy -= targets[t].T # backprop into y

      dWhy += np.dot(dy, hs[t].T)
      dby += dy

      # h[t] influences the cost in 5 ways:

      # through the interpolation using z at t+1
      dha = np.multiply(dhnext, ones - zs[t+1])

      # through transformation by weights into rbar
      dhb = np.dot(self.Rhr.T,drbarnext) 

      # through transformation by weights into zbar
      dhc = np.dot(self.Rhz.T,dzbarnext)

      # through transformation by weights into cbar
      dhd = np.multiply(rs[t+1],np.dot(self.Rhc.T,dcbarnext)) 

      # through the output layer at time t
      dhe = np.dot(self.Why.T,dy) 

      dh = dha + dhb + dhc + dhd + dhe
    
      dc = np.multiply(dh,zs[t]) 
      
      #backprop through tanh
      dcbar = np.multiply(dc , ones - np.square(cs[t])) 

      dr = np.multiply(hs[t-1],np.dot(self.Rhc.T,dcbar))
      dz = np.multiply( dh, (cs[t] - hs[t-1]) )

      # backprop through sigmoids
      drbar = np.multiply( dr , np.multiply( rs[t] , (ones - rs[t])) )
      dzbar = np.multiply( dz , np.multiply( zs[t] , (ones - zs[t])) )

      dWxr += np.dot(drbar, xs[t].T)
      dWxz += np.dot(dzbar, xs[t].T)
      dWxc += np.dot(dcbar, xs[t].T)

      dRhr += np.dot(drbar, hs[t-1].T)
      dRhz += np.dot(dzbar, hs[t-1].T)
      dRhc += np.dot(dcbar, np.multiply(rs[t],hs[t-1]).T)

      dbr += drbar
      dbc += dcbar 
      dbz += dzbar

      dhnext =    dh
      drbarnext = drbar 
      dzbarnext = dzbar 
      dcbarnext = dcbar

    deltas = [   dWxc
               , dWxr
               , dWxz
               , dRhc
               , dRhr
               , dRhz
               , dWhy
               , dbc
               , dbr
               , dbz
               , dby
             ]

    return loss, deltas, ps

