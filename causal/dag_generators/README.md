# DAG Generators
DAG Generation happens in two phases. The first phase is selecting which nodes
to include in the DAG, along with how to connect them. The second phase is 
actually initializing and creating those nodes.

The first phase is handled by subclasses of `DAGGenerator`. 
These will select a subset of given nodes listed in a YAML config file, and 
will choose certain edges connecting nodes to one another.

How those nodes and edges are selected is determined by a given `DAGGenerator`
subclass' specific algorithm, provided in the `select_nodes_and_edges` method.

# Creating Custom DAG Generation Methods

If you want to create your own algorithm for selecting which nodes and edges
to include in a DAG, create a subclass of the `DAGGenerator` class located in 
[dag_generator.py](dag_generator.py) and register it with the Factory 
(see [custom_dag.py](custom_dag.py) for how to do this). The subclass will need 
to implement the following methods:
- `select_nodes_and_edges`: Algorithm for selecting which nodes and edges to 
    include in the DAG.
- `ensure_valid_config`: Ensures that all of the necessary information is in the
    config file for creating the DAG from this method.

Then in your YAML file set the `dag_generation_method` to the name of this 
subclass.

# License
Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC

All rights reserved.

Distributed under the terms of the BSD 3-Clause License.
