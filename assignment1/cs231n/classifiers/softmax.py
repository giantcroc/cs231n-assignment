import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    deno = np.sum(np.exp(scores))
    loss = loss - scores[y[i]] + np.log(deno)

    for j in range(num_class):
      if j == y[i]:
        dW[:,j] += (np.exp(scores[y[i]]) - deno)/deno*X[i].T
      else:
        dW[:, j] += np.exp(scores[j])/deno*X[i].T
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  # Compute average
  loss /= num_train
  dW /= num_train

  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W).T
  scores -= np.max(scores,axis=0)
  deno = np.sum(np.exp(scores),axis=0)
  loss += np.sum(np.log(deno)) - np.sum(scores[y,np.arange(num_train)])

  mask = np.ones(scores.shape)/deno
  #print mask
  mask = mask*np.exp(scores)
  mask[y,np.arange(num_train)] -= 1

  dW = np.dot(X.T, mask.T)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  # Compute average
  loss /= num_train
  dW /= num_train

  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
  return loss, dW

