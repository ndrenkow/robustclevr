# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import yaml 
from copy import deepcopy
from ..causal_model import ModelBase
from ..registry import Factory

import numpy as np


def subset(lst1, lst2):
    '''
        Returns True if lst1 is a subset of lst2
    '''
    for x in lst1:
        if x not in lst2:
            return False
    return True


def filter_dict(fields, kwargs):
    '''
        Returns subset of kwargs to only the keys listed in fields
        
        Parameters:
            fields: The keys to subset kwargs to
            kwargs: The dictionary to subset
    '''
    return {field: kwargs[field] for field in fields}


@Factory.register('DAG')
class DAG(ModelBase):
    '''
        Class which initializes a DAG (Directed Acyclic Graph)
        from a YAML file, ultimately used as a CausalModel to sample
        corruption values used to render Blender Scenes.
    '''
    def __init__(self, config_file, seed=None, load=False, save_all=False):
        '''
            Parameters:
                config_file: YAML file to generate DAG from
                seed: Seed to use for reproducibility in sampling from DAG
                load: If the DAG should load a copy of a previously saved dag
                save_all: If all nodes in the DAG should generate an image
        '''

        with open(config_file, 'r') as f:
            self.configuration = yaml.safe_load(f)
        assert 'dag_generation_method' in self.configuration, 'Error! Config file must contain the "dag_generation_method"'

        self.loadable = load
       
        # dag_generation_method determines how to construct the DAG 
        # this amounts to selecting which nodes go in the DAG and 
        # which nodes should be connected to each other
        # i.e. selecting the nodes and edges of the DAG
        cls = Factory.create_class(self.configuration['dag_generation_method'], 
                config=self.configuration)
        nodes, edges = cls.select_nodes_and_edges()
        
        assert len(edges) > 0, 'Error! No edges in the graph'

        if seed is None:
            seed = int(np.random.randint(1e6))
        self.rng = np.random.default_rng(seed=seed)

        # construct the DAG with the selected nodes and edges
        self.initialize_nodes_and_edges(nodes, edges)

        # initialize as a causal model/dag
        super().__init__(save_all=save_all, seed=seed)


    def initialize_nodes_and_edges(self, nodes, edges):
        '''
            Given the nodes and edges selected from the config file,
            this will actually initialize and construct the DAG.

            Parameters:
                nodes: List of nodes to include in the DAG
                edges: List of edges to include in the DAG
        '''

        added_nodes = {}
        node_objects = {'root': None}
        
        # Create each node one by one
        while len(added_nodes) < len(nodes):
            added = False
            for child in nodes:
                child_name = child['name']
                child_parents = self.get_parents(child_name, edges)

                # Need to ensure that we initialize nodes in a top-down fashion
                # with respect to the dag structure, starting with the root
                # (note that "root" is not actually a node object)
                # So check we haven't initialized child and all parents initialized
                if child_name not in node_objects and subset(child_parents, node_objects):
                    node_params = self.get_node_parameters(child_name)
                    node_params['parents'] = [node_objects[parent] 
                            for parent in child_parents if parent != 'root']
                    
                    for field in ['type', 'corruption_func']:
                        assert field in node_params, f'Error in Node creation; "{field}" not given in config file for node "{child_name}"'

                    cls = node_params.pop('type')
                    corruption_func = node_params.pop('corruption_func')

                    # Initializing the Node
                    node_params['rng'] = self.rng
                    Factory.create_func(cls).ensure_valid_node(node_params)
                    necessary_fields = Factory.create_func(cls).necessary_fields
                    node_params = filter_dict(necessary_fields, node_params)
                    node = Factory.create_class(cls, **node_params)

                    # Allow for intervening on specific nodes
                    # Forcing it's sampled values to be of a specific 
                    # render value or severity
                    if 'intervene' in node_params:
                        # Get the intervening render value and severity value
                        if 'severity_value' in node_params['intervene']:
                            severity_value = node_params['intervene']['severity_value']
                            if 'render_value' in node_params['intervene']:
                                render_value = node_params['intervene']['render_value']
                            else:
                                render_value = float(node.get_render_value_from_severity(severity_value))
                        elif 'render_value' in node_params['intervene']:
                            render_value = node_params['intervene']['render_value']
                            severity_value = float(node.get_severity_from_render_value(render_value))
                        else:
                            assert False, 'Error! Must intervene on at least one of "severity_value" or "render_value"'

                        assert 0 <= severity_value <= 1, 'Error! Severity value must be between 0 and 1'

                        # Construct a ConstantNode which deterministically
                        # samples at a specific render and severity value
                        # using the determined intervening values
                        intervene_params = {}
                        for field in ['name', 'parameter', 'defaults']:
                            intervene_params[field] = node_params[field]
                        intervene_params['render_value'] = render_value
                        intervene_params['severity_value'] = severity_value
                        intervene_params['parents'] = node_params['parents']
                        cls = 'ConstantNode'
                        node_params = intervene_params

                        Factory.create_func(cls).ensure_valid_node(node_params)
                        necessary_fields = Factory.create_func(cls).necessary_fields
                        node_params = filter_dict(necessary_fields, node_params)
                        node = Factory.create_class('ConstantNode', **node_params)

                    # If loading specific values for the constructed node's
                    # internal parameters do so here
                    if self.loadable:
                        node_params = [params for params in self.configuration['node_list'] if params['name'] == child_name][0]
                        node.load(node_params)
                   
                    # Add the sampling and Blender corruption functions to
                    # the information associated with the node
                    node_info = {}
                    node_info['name'] = node.name
                    node_info['causal_func'] = node.causal_func
                    node_info['corruption_func'] = corruption_func

                    added_nodes[child_name] = node_info
                    node_objects[child_name] = node
                    added = True

            # If no nodes can be added 
            # (because they still have parents that still can't be added)
            # then there must be a cycle in the graph
            assert added, f'Error! Graph with edges {edges} has a cycle'

        # Setting parameters for the super classes' methods to work
        node_objects.pop('root')
        self.node_objects = node_objects
        self.node_list = list(added_nodes.values())
        self.edge_list = edges


    def get_node_parameters(self, name):
        '''
            Given the name of a node will pull it's parameters as specified
            in the config file.
        '''
        node_parameters = [deepcopy(node_params) for node_params in self.configuration['node_list'] if node_params['name'] == name][0]
        if 'defaults' not in node_parameters:
            node_parameters['defaults'] = {}
        return node_parameters


    @staticmethod
    def get_parents(node_name, edges):
        '''
            Given a node and the list of edges in a graph, will return 
            all parents of the node.
        '''
        parents = []
        for parent_name, child_name in edges:
            if child_name == node_name and parent_name not in parents:
                parents.append(parent_name)

        assert len(parents) > 0, f'Error! Node {node_name} has no parents'
        return parents


    def save(self, filepath):
        '''
            Will save the constructed DAG as a YAML file that can be used
            to load a copy of the DAG via this class' "load" method.

            Parameters:
                filepath: Full path and name of where to save the dag to
        '''
        assert filepath.endswith('.yaml') or filepath.endswith('.yml')

        # Get dictionary representation of graph along with original config
        # for reference in how it was constructed
        _yaml = self.to_yaml()
        _yaml['original_configuration'] = self.configuration

        # Forces any graph constructed from this YAML to select this graph's 
        # exact nodes and edges to ensure an exact copy can be loaded
        _yaml['dag_generation_method'] = 'CustomDAG' 
        _yaml['loadable'] = True
        _yaml['seed'] = self.seed
        with open(filepath, 'w') as f:
            yaml.dump(_yaml, f, default_flow_style=False)


    @classmethod
    def load(cls, config_file):
        '''
            Will create an instance of this class that loads the exact DAG
            specified in the config_file. This file should be generated from
            a previous instance's "save" method.

            Parameters:
                config_file: Full path and name of config file to load the DAG from
        '''
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        assert 'loadable' in config and config['loadable'], 'Error! Config file must be loadable. Save a previous DAG to achieve this.'
        assert 'dag_generation_method' in config and config['dag_generation_method'] == 'CustomDAG', 'Error! To load a DAG it must be a CustomDag.'
        assert 'seed' in config, 'Error! Seed must be in the config file!'
        
        return cls(config_file=config_file, load=True, seed=config['seed'])


    def to_yaml(self):
        '''
            Gets dictionary representation of DAG which can be saved and loaded
            using YAML.
            
            Will store each edge in the DAG along with each node and its 
            internal parameters.
        '''
        node_list = []

        # Get internal parameters of each node
        for node in self.node_objects.values():
            node_parameters = node.to_yaml()

            if len(node_parameters['defaults']) == 0:
                node_parameters.pop('defaults')

            # Add in Blender corruption function each node is associated with
            corruption_func = [n['corruption_func'] for n in self.node_list if n['name'] == node.name][0]
            node_parameters['corruption_func'] = corruption_func
            node_list.append(node_parameters)

        edge_list = self.edge_list

        return {'node_list': node_list, 'edge_list': edge_list}
