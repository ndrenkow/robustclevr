# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

from ..registry import Factory
from .node import Node


@Factory.register('ConstantNode')
class ConstantNode(Node):
    '''
        Node which will deterministically return a specific rendering and 
        severity value instead of randomly sampling these.
    '''
    necessary_fields = ['name', 
            'parents', 
            'parameter',
            'defaults',
            'render_value',
            'severity_value']

    def __init__(self,
        name, 
        parents,
        parameter,
        defaults,
        render_value,
        severity_value):
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

                render_value: The specific render value to corrupt images with
                severity: The specific severity (between 0 and 1) of 
                    these corruptions
        '''
        
        self.render_value = render_value
        self.severity_value = severity_value

        super().__init__('ConstantNode', name, parents, parameter, defaults)

    @classmethod
    def ensure_valid_node(cls, kwargs):
        '''
            Ensures that the node is being initialized with a specified severity 
            and render value.
        '''
        super().ensure_valid_node(kwargs)
        for field in cls.necessary_fields:
            assert field in kwargs, f'Error in ConstantNode creation; "{field}" not given in config file for node "{kwargs["name"]}"'

        assert 0 <= kwargs['severity_value'] <= 1

    def sampling_function(self, **parent_samples):
        '''
            Returns the predetermined render and severity value.
        '''
        return self.render_value, self.severity_value

    def to_yaml(self):
        '''
            Returns a dictionary of the node's internal parameters.
        '''
        _yaml = super()._to_yaml()
        _yaml['render_value'] = self.render_value
        _yaml['severity_value'] = self.severity_value
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
                _yaml: The dictionary storing the node's internal parameters
        '''
        self.render_value = _yaml['render_value']
        self.severity_value = _yaml['severity_value']

    def get_render_value_from_severity(self, severity):
        '''
            Returns the render value of the node.
        '''
        return self.render_value

    def get_severity_from_render_value(self, render_value):
        '''
            Returns the severity value of the node.
        '''
        return self.severity_value
