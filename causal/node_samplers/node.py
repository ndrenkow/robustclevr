# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

from collections import OrderedDict

class Node:
    '''
        Parent class of all Nodes that will be part of a DAG.

        Each subclass needs to implement:
            A to_yaml and load method, that will
            save and load a specific node's internal parameters to/from a dictionary.
            
            A get_severity_from_render_value and get_render_value_from_severity
            function which each map from a severity (between 0 and 1) to a
            specific rendering value (and vice versa)

            A sampling function, which will (randomly or deterministically)
            sample a severity and render value (used by Blender) for a given node

            necessary_fields class variable, which is all of the fields needed
            during initialization per the __init__ method.
    '''
    necessary_fields = ['name', 
            'parents', 
            'parameter', 
            'defaults']

    def __init__(self, 
            node_type,
            name, 
            parents, 
            parameter, 
            defaults):
        '''
            Initialize a general Node.

            Parameters:
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
        '''

        self.node_type = node_type
        self.name = name
        self.parents = parents
        self.parameter = parameter
        self.defaults = defaults

    def causal_func(self, **parent_samples):
        '''
            The causal function of the node as needed by the overall DAG the
            node is a part of.

            This will call the node's sampling function to get a severity
            and rendering value to corrupt an image with and update a dictionary
            storing all node's sampled values.

            Parameters:
                parent_samples: Previous results of this function call for the 
                    parents of this node. Stores the parent's sampled 
                    severities and render values, allowing for child nodes
                    to have their sampling behavior affected by parents'. This
                    allows for a Causal Mechanism for sampling corruptions.
        '''
        render_value, severity_value = self.sampling_function(**parent_samples)
        samples = OrderedDict()
        samples[self.parameter] = render_value
        samples['severity'] = severity_value
        samples['sampled'] = self.parameter
        for param, default in self.defaults.items():
            samples[param] = default

        return samples

    @classmethod
    def ensure_valid_node(cls, kwargs):
        '''
            Ensures that the arguments associated with a node as specified in, 
            or determined by a config file contain the minimum necessary 
            arguments common to all nodes.

            This method should be implemented for subclasses as well, as they
            may require more specific fields or values, but each subclass 
            should also invoke this parent classes' method to ensure that it 
            meets the requirements common to all nodes.
        '''
        for field in cls.necessary_fields:
            assert field in kwargs, f'Error in Node creation; "{field}" not given in config file for node "{kwargs["name"]}"'

    def sampling_function(self):
        raise NotImplementedError

    def save(self):
        return self.to_yaml()

    def load(self, _yaml):
        raise NotImplementedError

    def to_yaml(self):
        raise NotImplementedError

    def _to_yaml(self):
        '''
            Returns information associated with each node.

            Each subclass' to_yaml method should invoke this _to_yaml method.
        '''
        return {'name': self.name,
                'parameter': self.parameter,
                'defaults': self.defaults,
                'type': self.node_type,
                }

    def get_severity_from_render_value(self, render_value):
        raise NotImplementedError

    def get_render_value_from_severity(self, severity):
        raise NotImplementedError
    
    def __str__(self):
        return f'{self.node_type} "{self.name}" with parents {[parent.name for parent in self.parents]}'

    def __repr__(self):
        return self.__str__()
