# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

from copy import deepcopy

class DAGGenerator:
    '''
        Parent class for generating DAGs (as used in ModelGenerator class above).

        Subclasses of this class will implement methods to select nodes
        and edges from a config file in order to construct a DAG.
    '''
    def __init__(self, config):
        self.config = deepcopy(config)
        self.ensure_valid_config()

    def select_nodes_and_edges(self):
        raise NotImplementedError

    def ensure_unique_names(self):
        '''
            Ensures that each node in the config file has a unique name.
        '''
        names = []
        for node in self.config['node_list']:
            assert 'name' in node, 'Error! Each node in the node list must have a name'
            assert node['name'] != 'root', 'Error! Cannot have "root" as a node name'
            assert node['name'] not in names, f'Error! Multiple nodes with name "{node["name"]}"'
            names.append(node['name'])

    def ensure_valid_config(self):
        '''
            Ensures that a config file is valid for this DAGGenerator.

            Note that each DAGGenerator subclass may require its own specific
            values and assumes different structures or keys are present in the
            config file, so each subclass should implement its own 
            "ensure_valid_config" method, but should also call this one, which
            checks for validity constraints universal to all DAGs.

            (checking for cycles in a DAG is done during DAG Generation within
            the ModelGenerator class above).
        '''
        assert 'node_list' in self.config, 'Error! Config file must contain "node_list"'
        self.ensure_unique_names()
