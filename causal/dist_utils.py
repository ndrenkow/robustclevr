# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import torch
import torch.distributions as td


class DiscreteUniform:
    """ Returns quantized values in [0, 1) where quantization is determined by n_bins.
    Quantization ensures non-zero probability of sampling values within range. 
    """
    def __init__(self, n_bins):
        self.dist = td.Categorical(torch.ones(n_bins) * (1./n_bins))
        self.n_bins = n_bins

    def sample(self):
        val = self.dist.sample() 
        logprob = self.dist.log_prob(val)
        return val.float().item() / self.n_bins, logprob.item()


class Uniform:
    """ Returns uniform values in [low, high) or [0, 1). Note samples have probability 0 """
    def __init__(self, low=None, high=None):
        low = low if low is not None else 0.
        high = high if high is not None else 1.
        self.dist = td.Uniform(low, high)

    def sample(self):
        val = self.dist.sample()
        logprob = self.dist.log_prob(val)
        return val.item(), logprob.item()


class Categorical:
    """ Returns discrete choice within range """
    def __init__(self, n_bins=None, probs=None, logits=None):
        if n_bins is not None:
            probs = torch.ones(n_bins) / float(n_bins)
            self.dist = td.Categorical(probs=probs)
        elif probs is not None:
            self.dist = td.Categorical(probs=probs)
        else:
            self.dist = td.Categorical(logits=logits)

    def sample(self):
        val = self.dist.sample() 
        logprob = self.dist.log_prob(val)
        val = val
        return val.float().item(), logprob.item()


class HalfNormal:
    """ Returns a half normal distribution with specified scale """
    def __init__(self, scale=1):
        self.dist = td.HalfNormal(scale=scale)

    def sample(self):
        val = self.dist.sample()
        logprob = self.dist.log_prob(val)
        return val.item(), logprob.item()


class Normal:
    """ Returns a half normal distribution with specified scale """
    def __init__(self, loc=0, scale=1):
        self.dist = td.Normal(loc=loc, scale=scale)

    def sample(self):
        val = self.dist.sample()
        logprob = self.dist.log_prob(val)
        return val.item(), logprob.item()


class Bernoulli:
    def __init__(self, p):
        self.dist = td.Bernoulli(torch.tensor([p]))
    
    def sample(self):
        val = self.dist.sample()
        logprob = self.dist.log_prob(val)
        return val.item(), logprob.item()

