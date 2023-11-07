import torch
from torch import nn

class CRF(nn.Module):

  def __init__(self, labels):
    raise NotImplementedError()
    
  # compute the potential function for a given state and observation
  def compute_potential():
    raise NotImplementedError()

  # decode the target sequence prediction
  def viterbi():
    raise NotImplementedError()

  # Compute the gradient of the log-likelihood
  def gradient():
    raise NotImplementedError()
