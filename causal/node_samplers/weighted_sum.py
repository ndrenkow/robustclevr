# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import sys 

import numpy as np
from scipy.special import betaincinv
from ..registry import Factory
from .node import Node


@Factory.register('WeightedSumNode')
class WeightedSumNode(Node):
    '''
        Node which will randomly sample a severity and render value based on
        a linear combination of the severities of its parent nodes, along
        with some added noise and a non-linearity.

        The distribution of the node's render value can be specified to follow
        the shape of any beta distribution, allowing for more flexibilty 
        in distributions of sampled severity and corruption values.
    '''
    necessary_fields=['name', 
        'parents', 
        'parameter',
        'defaults',
        'min_val', 
        'max_val', 
        'extreme', 
        'standard', 
        'beta_a', 
        'beta_b', 
        'corruption_type', 
        'bias', 
        'std', 
        'activation_type',
        'rng']

    def __init__(self, 
        name, 
        parents, 
        parameter,
        defaults,
        min_val, 
        max_val, 
        extreme, 
        standard, 
        beta_a, 
        beta_b, 
        corruption_type, 
        bias, 
        std, 
        activation_type,
        rng):
        '''
            Initialize a ConstantNode.

            Parameters:
                As needed by Node superclass:

                node_type: The subclass/specific node class
                name: The name of the node (as specified in a config file)
                parents: A list of Node objects which are the parents of
                    this node in the DAG
                parameter: The name of the parameter this node is sampling
                    a render value for in the "corruption_func" associated 
                    with the node
                defaults: A dictionary storing the parameters and render values 
                    for the node's associated "corruption_func" which will be
                    deterministically set instead of (potentially) randomly 
                    sampled like the node's parameter.

                As needed by the ConstantNode class:

                min_val: The minimum allowed render value for corruptions
                max_val: The maximum allowed render value for corruptions
                extreme: The render value corresponding to the most extreme
                    image corruption
                standard: The render value corresponding to the least extreme
                    image corruption (often times is no corruption at all)
                beta_a: The a value for the beta distribution of the node, where
                    the render value (on a 0-1 scale) follows a Beta(a,b) 
                    distribution (assuming there is no causal influence from
                    parents)
                beta_b: The b value for the beta distribution of the node, where
                    the render value (on a 0-1 scale) follows a Beta(a,b) 
                    distribution (assuming there is no causal influence from
                    parents)
                corruption_type: What kind of corruption this is. Can be one of
                    increasing: The larger the render value the more severe the corruption
                    decreasing: The smaller the render value the more sever the corruption
                    centered: The farther a render value is from some nominal value,
                        (in either direction) the more severe the corruption
                bias: Mean of random noise used during sampling of severity and
                    render value
                std: Standard deviation of random noise used during sampling
                    of severity and render value
                activation_type: Non-linear activation function with a range
                    of [0, 1] used during sampling of severity and render value
                rng: RNG object used for random sampling. Allows for 
                    reproducibility if desired
        '''

        self.min_val = min_val
        self.max_val = max_val
        self.extreme = extreme
        self.standard = standard
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.corruption_type = corruption_type
        self.std = std
        self.activation_type = activation_type
        self.set_activation()
        self.rng = rng

        if isinstance(bias, float) or isinstance(bias, int):
            self.bias = float(bias)
        elif bias == True or bias == "random":
            self.bias = self.rng.normal(0, 1)
        else:
            self.bias = 0.

        # sample edge weights from parents
        self.edge_weights = {}
        for parent in parents:
            self.edge_weights[parent.name] = self.rng.uniform(-1, 1)

        super().__init__('WeightedSumNode', name, parents, parameter, defaults)


    @classmethod
    def ensure_valid_node(cls, kwargs):
        '''
            Ensures that the node has all the specific parameters needed for its
            initialization, and that certain criteria are met for each parameter.
        '''
        super().ensure_valid_node(kwargs)
        for field in cls.necessary_fields:
            assert field in kwargs, f'Error in WeightedSumNode creation; "{field}" not given in config file for node "{kwargs["name"]}"'

        assert kwargs['min_val'] < kwargs['max_val']
        assert kwargs['corruption_type'] in ['increasing', 'decreasing', 'centered']
        assert kwargs['beta_a'] > 0
        assert kwargs['beta_b'] > 0
        assert kwargs['std'] > 0
        assert kwargs['activation_type'] in ['sigmoid', 'tanh']


    def set_activation(self):
        '''
            Sets the non-linear activation function of the node to either
            the sigmoid function or hyperbolic tangent function (which is 
            mapped onto a [0, 1] range)
        '''
        if self.activation_type == 'sigmoid':
            self.activation_fn = lambda x: 1 / (1 + np.exp(-x))
        elif self.activation_type == 'tanh':
            self.activation_fn = lambda x: (np.tanh(x) + 1) / 2
        else:
            sys.exit(f'Error: invalid activation function {self.activation_type}')


    def sampling_function(self, **parent_samples):
        '''
            Randomly samples a severity and render value to corrupt an image 
            with.

            Takes a weighted combination of parent's sampled severities, and 
            adds random noise.
            This is then mapped to a [0, 1] range via the non-linear activation
            function and represents a quantile.
            Taking the inverse of the Beta distribution associated with this node
            gives a value between 0 and 1 which represents the rendering value
            to corrupt the image with, which then has to be scaled back to the
            range of [min_val, max_val].

            The quantile approach of the Beta distribution allows for sampling
            from a wider variety of distributions than a uniform distribution.
        '''
        # Weighted sum of parents severities
        total = 0
        for parent in self.parents:
            total += self.edge_weights[parent.name] * parent_samples[parent.name]['severity']
        
        # Add noise 
        total += self.rng.normal(self.bias, self.std)

        # Turn into quantile and get render value (on 0-1 scale) from beta distribution
        quantile = self.activation_fn(total)
        x = betaincinv(self.beta_a, self.beta_b, quantile)

        # undo minmax scale to get true render value
        render_value = self.min_val + x * (self.max_val - self.min_val)

        # calculate severity
        severity = self.get_severity_from_render_value(render_value)

        return render_value, severity


    def get_severity_from_render_value(self, render_value):
        '''
            Calculates the severity of a corruption based on the render value
            used in the corruption.
        '''
        # severity between 0 and 1
        if self.corruption_type == 'increasing':
            severity = (render_value - self.min_val) / (self.max_val - self.min_val)
        elif self.corruption_type  == 'decreasing':
            severity = (self.max_val - render_value) / (self.max_val - self.min_val)
        elif self.corruption_type  == 'centered':
            distance = np.abs(render_value - self.standard)
            max_distance = max(np.abs(self.max_val - self.standard), np.abs(self.min_val - self.standard))
            severity = (distance / max_distance)
        else:
            sys.exit('Error: node\'s corruption_type must be one of "increasing", "decreasing", or "centered", not "{}"'.format(self.corruption_type))
        return severity


    def get_render_value_from_severity(self, severity):
        '''
            Calculates a render value to corrupt an image with at a given
            severity.
        '''
        if self.corruption_type == 'increasing':
            render_value = self.min_val + (severity * (self.max_val - self.min_val))
        elif self.corruption_type == 'decreasing':
            render_value = self.min_val + ((1 - severity) * (self.max_val - self.min_val))
        elif self.corruption_type == 'centered':
            max_distance = max(np.abs(self.max_val - self.standard), np.abs(self.min_val - self.standard))
            distance = severity * max_distance
            render_value = self.standard + (distance * (self.rng.integers(2)*2 - 1))   # normal +/- distance (+/- chosen randomly)
        else:
            sys.exit('Error: node\'s corruption_type must be one of "increasing", "decreasing", or "centered", not "{}"'.format(self.corruption_type))
        return render_value


    def to_yaml(self):
        '''
            Returns a dictionary of the node's internal parameters.
        '''
        _yaml = super()._to_yaml()
        _yaml['min_val'] = self.min_val
        _yaml['max_val'] = self.max_val
        _yaml['extreme'] = self.extreme
        _yaml['standard'] = self.standard
        _yaml['beta_a'] = self.beta_a
        _yaml['beta_b'] = self.beta_b
        _yaml['corruption_type'] = self.corruption_type
        _yaml['std'] = self.std
        _yaml['activation_type'] = self.activation_type
        _yaml['bias'] = self.bias
        _yaml['edge_weights'] = self.edge_weights
        return _yaml

    def save(self):
        '''
            Returns a dictionary of the node's internal parameters.
        ''' 
        return self.to_yaml()

    def load(self, _yaml):
        '''
            Sets the node's internal parameters to those specified as input.

            Parameters:
                _yaml: The dictionary storing the node's internal parameters.
        '''
        self.min_val = _yaml['min_val']
        self.max_val = _yaml['max_val']
        self.extreme = _yaml['extreme']
        self.standard = _yaml['standard']
        self.beta_a = _yaml['beta_a']
        self.beta_b = _yaml['beta_b']
        self.corruption_type = _yaml['corruption_type']
        self.std = _yaml['std']
        self.activation_type = _yaml['activation_type']
        self.bias = _yaml['bias']
        self.edge_weights = _yaml['edge_weights']
