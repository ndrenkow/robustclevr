# Nodes
DAG Generation happens in two phases. The first phase is selecting which nodes
to include in the DAG, along with how to connect them. The second phase is 
actually initializing and creating those nodes.

These files relate to the second phase. 

Each node is associated with a specific corruption, and different Node types 
have different algorithms which will (either randomly or deterministically) 
sample a severity and render value to corrupt an image with.

Depending on the kind of algorithm you want to use to sample these corruption
values, you may want some nodes to be of one type, while other are of a 
different type.

# Creating Custom Nodes

If you want to create your own algorithm for how to sample a severity and 
render value to corrupt an image with, create a subclass of the `Node` class 
located in [node.py](node.py) and register it with
the Factory (see [constant.py](constant.py) for how to do this). The subclass 
will need to implement the following methods:
- `__init__`: How to initialize the node
- `ensure_valid_node`: Checking that all of the parameters necessary to 
    initialize the node are given.
- `sampling_function`: The algorithm for calculating a severity and render value
    to corrupt an image with.
- `to_yaml`: A function that will return a dictionary storing the node's 
    internal parameters.
- `load`: A function which will set a node's internal parameters to those 
    provided in a dictionary.
- `get_severity_from_render_value`: Given a render value, it will calculate the 
    severity of corrupting an image with that value.
- `get_render_value_from_severity`: Given a severity, it will determine the 
    render value to corrupt an image with at that severity.

Then in your YAML file set a Node's type to be the name of this subclass.

# Intervening on a Node

Optionally nodes can also be intervened on, so that they are forced to sample
a specific rendering or severity value. To do this, add the `intervene` field
to the node's configuration in the config file and specify what you want to 
intervene on, and with what value. 

For example, the configuration file 
[example1.yaml](../graph_configs/example1.yaml) can be forced to have the 
`defocus` node always sample with an `f_stop` value of 75
```
dag_generation_method: "CustomDAG"
edge_list:
    - ["root", "G"]
    - ["G", "D"]
node_list:
    - name: "G"
      corruption_func: "gamma"
      parameter: "gamma"
      type: "ConstantNode"
      render_value: 1.0
      severity_value: 0.6

    - name: "D"
      corruption_func: "defocus"
      parameter: "f_stop"
      type: "WeightedSumNode"
      defaults:
        z: 1.0
      min_val: 50
      max_val: 200
      extreme: 50
      standard: 200
      beta_a: 1
      beta_b: 1
      corruption_type: "decreasing"
      bias: False
      std: 0.5
      activation_type: "tanh"
      intervene:
        render_value: 75
```

or it can be forced to always render with a `severity` of 0.8
```
dag_generation_method: "CustomDAG"
edge_list:
    - ["root", "G"]
    - ["G", "D"]
node_list:
    - name: "G"
      corruption_func: "gamma"
      parameter: "gamma"
      type: "ConstantNode"
      render_value: 1.0
      severity_value: 0.6

    - name: "D"
      corruption_func: "defocus"
      parameter: "f_stop"
      type: "WeightedSumNode"
      defaults:
        z: 1.0
      min_val: 50
      max_val: 200
      extreme: 50
      standard: 200
      beta_a: 1
      beta_b: 1
      corruption_type: "decreasing"
      bias: False
      std: 0.5
      activation_type: "tanh"
      intervene:
        severity_value: 0.8
```

# License
Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC

All rights reserved.

Distributed under the terms of the BSD 3-Clause License.
