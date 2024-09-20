# RobustCLEVR - A Benchmark and Framework for Evaluating Robustness in Object-centric Learning
This repository provides a framework to generate images and scenes with a wide 
range of imaging conditions generated with Blender.  These conditions are sampled
according to causal models that encode interactions between factors of the 
environment (e.g., weather, lighting), sensor (e.g., f-stop, exposure, ISO), 
and processing pipeline (e.g., image compression, auto white-balance).  This 
repository was the basis for the paper ["RobustCLEVR: A Benchmark and Framework for 
Evaluating Robustness in Object-centric Learning"](https://openaccess.thecvf.com/content/WACV2024/html/Drenkow_RobustCLEVR_A_Benchmark_and_Framework_for_Evaluating_Robustness_in_Object-Centric_WACV_2024_paper.html).

# Installation
Clone this repository and install a working version of 
[Blender 3.x python](https://builder.blender.org/download/bpy/) on your system.

Install the necessary packages used by the causal framework in this repository. 
These can be found in [causal_dependencies.txt](causal/causal_dependencies.txt).

To use Blender's version of python with these packages, you will need to install 
these packages into the python that Blender is shipped with.

## Installing Packages in Blender's Python
Run the following steps (with potentially a different version of python if you 
have one)
```
<path to your blender>/python/bin/python3.10 -m ensurepip
<path to your blender>/python/bin/python3.10 -m pip install --upgrade pip
<path to your blender>/python/bin/python3.10 -m pip install -r causal_dependencies.txt
```

### Compatible Python Versions
We have tested this installation process and repository with python3.10, but 
other versions of python may also work.

# Creating a Causal Model
Causal Models are visually represented as Directed Acyclic Graphs (DAGs) where each node
is tied to an imaging setting/corruption, and directed edges of the DAG indicate 
a causal relationships between nodes (e.g., `Lighting --> Exposure`).

DAG generation occurs in the [dag_generators](causal/dag_generators) directory. 

## DAG Generation
The `DAG` class in [dag_generators/dag.py](causal/dag_generators/dag.py) will create a 
DAG and through the `ModelBase` class in [causal_model.py](causal/causal_model.py) it will
be able to interact with Blender to render images. 

DAG Generation is handled entirely from a single YAML file.

To create a DAG you need only create an instance of the `DAG` by 
passing in a config file 
```
causal_model = DAG("graph_configs/example1.yaml")
```

An example YAML file may look like:
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
```

Each YAML file needs to include a `dag_generation_method` and a `node_list`.

The `dag_generation_method` indicates which Class to use to determine which 
nodes and edges to include in the DAG. In the example above, the `CustomDAG`
is used and creates the DAG exactly as specified in the YAML file. 

If you want to create your own `dag_generation_method`, which may build a DAG
from randomly selecting some nodes or edges, or via any other means, refer to
the `README` in the [dag_generators](causal/dag_generators) directory.

The `node_list` specifies which nodes can be selected to be a part of the DAG.
Each node needs at a minimum to include the following fields:
- `name`: The name of the node
- `corruption_func`: This is the name of the function as specified in 
    [corruptions_compositor.py](causal/corruptions_compositor.py) which will actually
    apply a corruption to a blender scene/image.
- `parameter`: Some corruption functions (like `defocus`) have multiple 
    parameters that Blender can change to corrupt an image. This is used to
    specify which parameter the value applies to.
- `type`: The name of the class that this node is an instance of. Each class
    has its own algorithm for determining how it should sample its corruption.
    For example, the `WeightedSumNode` determines the corruption value based
    on a weighted combination of its parents sampled values.
    In contrast, the `ConstantNode` always sets the corruption to a specific
    value.

If you want to create your own `Node` class, which determines how to sample 
severity and corruption values based on your own algorithm, refer to the 
`README` in the [node_samplers](causal/node_samplers) directory.

Depending on the type of the node, other values may need to be specified as 
well. For example, the `ConstantNode` needs to also be given the `render_value` 
and `severity_value` field.

## Intervening on Nodes
You can also intervene on specific nodes in order to force them to have specific
severities or rendering values. To do this, add in the `intervene` option to a 
node(s) in a config file, and then specify how you want to intervene.
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
You can either intervene on the `severity_value` or the `render_value`. The 
`severity_value` refers to how severe the corruption is, and the `render_value`
refers to the actual value used to set Blender's internal settings to when
rendering the scene.

As such, a `render_value` can potentially take on any value, as long as Blender
allows for the corresponding parameter to take on such a value. In contrast, the
`severity_value` must be between `0` and `1` inclusive, where `0` is the least
severe corruption, and `1` is the most severe corruption.

## Saving and Loading DAGs
You can also save and load DAGs. To save, simply call the model's `save`
method
```
causal_model.save("saved_dags/my_dag.yaml")
```

Then you can load that exact same DAG with the load method
```
causal_model = DAG.load("saved_dags/my_dag.yaml")
```

## Wrapping Multiple DAGs
You can also create multiple DAGs and wrap them in a single 
`WrapperModel`. To do this, pass in the list of classes to create along 
with the list of config files
```
dag_classes = ["DAG", "DAG"]
config_files = ["graph_configs/example1.yaml", "graph_configs/example2.yaml"]
wrapped_models = WrapperModel(dag_classes=dag_classes, config_files=config_files)
```

The `WrapperModel` can be saved just like a regular `DAG`, and it
can load multiple `DAG` objects as well. This allows you to easily sample from
multiple DAGs to achieve different appearance properties for the 
same Blender scene.


# Creating Corrupted Scenes
To generate a blender image, one needs to first have already 
created a Blender scene by populating it with various objects. Here we use the code
from the original [the CLEVR repository](https://github.com/facebookresearch/clevr-dataset-gen)
for constructing the undistorted scenes.

Once the scene geometry is specified, call `render_corrupted_scene` which will 
use the `DAG` object to randomly sample values according to the causal model 
and use those values to render the final image.

The `render_corrupted_scenes` function can also add information about the 
sampled corruption values for each image to a dictionary that you pass in if 
you want to maintain that information. To do so, pass in the dictionary to the 
`scene_info` argument.

There is also an argument called `modes` which allows you to specify the types 
of render settings you want to sample. 

Passing in the `default` mode will render the scene using all nominal settings 
```
from render_utils import render_corrupted_scene
scene_info = {}
render_corrupted_scene(causal_model, modes=['default'], scene_info=scene_info)
```

If you do want to sample more challenging conditions, then you can pass in 
`random` to the `modes` argument, which will randomly sample render settings 
based on your Causal Model
```
render_corrupted_scene(causal_model, modes=['random'], scene_info=scene_info)
```

Note that `modes` expects a list of modes, so you can pass in more than one mode 
if you wish.

# Creating New Blender Corruptions
The compositor functionality of blender allows for specifying corruptions 
applied after rendering.  We provide a set of compositor-based corruptions in 
[corruptions_compositor.py](causal/corruptions_compositor.py) file adapted from 
(Hendrycks et al., 2019) [paper](https://arxiv.org/abs/1903.12261), 
[code](https://github.com/hendrycks/robustness/tree/master).

If you wish to add a new custom corruption you need to create a new function
in this file and register it with the Factory. Then set the `corruption_func` 
field of a node in the config file to the name of this new function. The 
parameter of that node should be the same name as one of the parameters the 
function takes in according to its signature.

# Notes
There are some Causal Models/DAGs which were created outside of the 
`DAG` class and methods as defined in this README. Those can be
found in [custom_scm/chain_model.py](causal/custom_scm/chain_model.py) and 
[custom_scm/tree_model.py](causal/custom_scm/tree_model.py).
These are provided to show how to create alternative custom causal models. 
In this way you can create a custom function for each node in 
the DAG and specify the DAG structure explicitly.

In order to create one of these you can either create it directly, or wrap it 
in a `WrapperModel`.
```
causal_model = TreeModel() # creating directly
wrapped_model = WrapperModel(dag_classes=["TreeModel"])
```

If you want to intervene on a node with one of these handcrafted DAGs, because 
they do not correspond to any YAML file you can instead intervene when rendering 
the scene.
```
render_corrupted_scene(causal_model, modes=['random'], scene_info=scene_info, intervene={"G": {"gamma": 2.4}})
```

Note that you can also intervene on multiple nodes at the same time.
```
render_corrupted_scene(causal_model, modes=['random'], scene_info=scene_info, intervene={"G": {"gamma": 2.4}, "N": {"factor": 0.1}})
```

This method of intervening can also be used for models created with YAML files
as well, though if possible it is advised to do so in the YAML files so it is 
better documented how scenes were generated.


# License
Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC

All rights reserved.

Distributed under the terms of the BSD 3-Clause License.


# Citation
```
@inproceedings{drenkow_robustclevr_2024,
  title={RobustCLEVR: A Benchmark and Framework for Evaluating Robustness in Object-centric Learning},
  author={Drenkow, Nathan and Unberath, Mathias},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={4518--4527},
  year={2024}
}
```