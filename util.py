import numpy as np

def gradCheck(model, deltas, inputs, targets, epsilon, tolerance):

  diffs = getDiffs(model, deltas, inputs, targets, epsilon)
  answer = True

  for diffTensor, name, delta in zip(diffs, model.names, deltas):

    if np.abs(diffTensor.max()) >= tolerance:
      print "DIFF CHECK FAILS FOR TENSOR: ", name
      print "DIFF TENSOR: "
      print diffTensor
      print "NUMERICAL GRADIENTS: "
      # diff = grad - delta => diff+delta = grad
      print diffTensor + delta
      print "BPROP GRADIENTS: "
      print delta
      answer = False
    else:
      pass

  return answer

def getDiffs(model, deltas, inputs, targets, epsilon):
  """
  For every (weight,delta) combo in zip(weights, deltas):
    Add epsilon to that weight and compute the loss (first_loss)
    Remove epsilon from that weight and compute the loss (second_loss)
    Check how close (first loss - second loss) / 2h is to the delta from bprop
  """

  diff_tensors = []
  for D in deltas:
    diff_tensors.append(np.zeros_like(D))

  for W,D,N,diffs in zip(model.weights, deltas, model.names, diff_tensors):
  # for each weight tensor in our model

    for i in range(W.shape[0]):
      for j in range(W.shape[1]):
        # for each weight in that tensor

        # compute f(x+h) for that weight
        W[i,j] += epsilon
        loss, ds, os = model.lossFun(inputs, targets)
        loss_plus = np.sum(loss)

        # compute f(x - h) for that weight
        W[i,j] -= epsilon*2
        loss, ds, os = model.lossFun(inputs, targets)
        loss_minus = np.sum(loss)

        # grad check must leave weights unchanged
        # so reset the weight that we changed
        W[i,j] += epsilon

        # compute the numerical grad w.r.t. this param
        grad = (loss_plus - loss_minus) / (2 * epsilon) 
        diffs[i,j] = grad - D[i,j] 

  return diff_tensors 
