# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

from .dag_generator import DAGGenerator
from ..registry import Factory


@Factory.register('CustomDAG')
class CustomDAG(DAGGenerator):
    '''
        Subclass of DAGGenerator which will select all nodes and edges 
        specified in the config file to be in the DAG.

        Each node will be constructed with the parameters specified in the 
        config file.
    '''
    def select_nodes_and_edges(self):
        '''
            Returns all nodes and edges specified in the config file to be
            in the DAG.
        '''
        nodes = []
        edges = []
        for node in self.config['node_list']:
            nodes.append(node)
        for edge in self.config['edge_list']:
            edges.append(edge)
        return nodes, edges

    def ensure_valid_config(self):
        '''
            Checks that an edge list is specified in the config file and that
            each node in the config file appears in the edge list, and that 
            each node in the edge list is provided in the node list.

            i.e. there are no unused nodes, nor any unspecified nodes in the
            edge list
        '''
        super().ensure_valid_config()

        assert 'edge_list' in self.config, 'Error! Config file must contain "edge_list"'

        node_names = [node['name'] for node in self.config['node_list']] + ['root']
        for edge in self.config['edge_list']:
            for node in edge:
                assert node in node_names, f'Error! No node with name "{node}" in node_list'

        for node in node_names:
            appears = False
            for edge in self.config['edge_list']:
                if node in edge:
                    appears = True
                    break
            assert appears, f'Error! Node "{node}" not in the edge list'
