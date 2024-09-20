# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import networkx as nx
from collections import OrderedDict
from sklearn.utils import Bunch
from box import Box
from .registry import Factory


class ModelBase:
    '''
        Parent class connecting initialized DAGs to image corruptions with
        Blender
    '''
    node_list = []
    edge_list = []
    
    def __init__(self, save_all=False, seed=1):
        '''
            Initialize a Causal Model. Relies on the DAGGenerator class
            having already initialized self.node_list and self.edge_list

            Parameters:
                save_all: Whether or not to save images corresponding to every
                    corruption in the DAG, or just the final output image
                    resulting from all corruptions collectively
                seed: Random seed used for reproducibility in sampling
        '''
        config = Box(dict(nodes=self.node_list, edges=self.edge_list))
        self.config = Box(config)
        self.save_all = save_all

        # self.nodes maps a node's name to it's sampling function
        # self.node_funcs maps a node's name to a function used to actually
        # corrupt the image (via blender)
        self.nodes = {}
        self.node_funcs = {}
        for n in self.config.nodes:
            n = Bunch(**n)
            self.nodes[n.name] = n.causal_func
            self.node_funcs[n.name] = n.corruption_func if 'corruption_func' in n else n.causal_func
            
        self.edges = self.config.edges
        self.observed_variables = frozenset(self.nodes)

        # construct DAG (directed acyclic graph)
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(self.nodes.keys())
        self.dag.add_edges_from(self.edges)

        assert nx.is_directed_acyclic_graph(self.dag)
        self.graph = self.dag.to_undirected()

        self.seed = seed

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.observed_variables)))
        return "{}({})".format(self.__class__.__name__, variables)

    def sample(self, scene=None, intervene={}, rng=None):
        '''
            Walks through DAG and samples severity and render value to corrupt
            image with in a top-down fashion.

            Note that sampled severities and render values of child nodes
            may depend on severities and render values of parent nodes, hence
            the walking through in a top-down fashion.

            Returns the sampled severity and render value to corrupt image with
            for each node in the DAG.
        '''
        node_params = OrderedDict()
        for node_name in nx.topological_sort(self.dag):
            if node_name.lower() == 'root':
                continue
            node_params[node_name] = self.nodes[node_name](**node_params, rng=rng)
            if node_name in intervene:  # still do above to maintain random state
                node_params[node_name].update(intervene[node_name])
        return node_params

    def update_tree(self, root_socket, tree, params, fn_base):
        '''
            Updates Blender's node tree to build corruptions as part of the 
            rendering process based on the corruption values provided in 
            params (the output of sample function above)

            Parameters:
                root_socket: Root node of Blender's rendering tree
                tree: Blender's active scene tree used in rendering the scene
                params: Dictionary storing sampled rendering values to corrupt
                    the image with for each node
                fn_base: Base of the filename for where to save output 
                    corrupted images
        '''
        # Creates new node sockets to be used in Blender's active scene tree
        # which will corrupt the image with the sampled render values 
        # specified in params
        blender_node_sockets = {node_name: Factory.create_func(self.node_funcs[node_name])(tree, **params[node_name])
                                for node_name in self.nodes}
        src_keys = list(blender_node_sockets.keys())
        for n in src_keys:
            # Create output image if it is a leaf node (final corruption)
            # or if save_all is set to True
            if self.save_all or len(nx.descendants(self.dag, n)) == 0:
                blender_node_sockets['{}-out'.format(n)] = Factory.create_func('output_file')(tree, fn_base, n)
                tree.links.new(blender_node_sockets[n].outputs['Image'], blender_node_sockets['{}-out'.format(n)].inputs['Image'])

        # Link each of the blender node sockets together in the structure 
        # specified by the DAG
        # Ensures that when rendering the image, the actual corruption values
        # and node sockets applying them are actually used
        for (src, dst) in self.dag.edges:
            if src == 'root':
                dst_node = blender_node_sockets[dst][0] if isinstance(blender_node_sockets[dst], list) else blender_node_sockets[dst]
                tree.links.new(root_socket, dst_node.inputs['Image'])
                continue
            src_node = blender_note_sockets[src][-1] if isinstance(blender_node_sockets[src], list) else blender_node_sockets[src]
            dst_node = blender_node_sockets[dst][0] if isinstance(blender_node_sockets[dst], list) else blender_node_sockets[dst]
            tree.links.new(src_node.outputs['Image'], dst_node.inputs['Image'])
