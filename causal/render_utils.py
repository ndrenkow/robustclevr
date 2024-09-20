# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import logging, sys
import bpy

sys.path.insert(0, '.')
from . import WrapperModel


def reset_tree(tree, initial_nodes):
    '''
    Resets the blender context tree to only the initial nodes

    Parameters:
        tree: The blender scene's active context tree to reset
        initial_nodes: The nodes to keep (initial ones in the tree)
    '''
    for node in tree.nodes:
        if node.name not in initial_nodes:
            tree.nodes.remove(node)

def render_corrupted_scene(model, 
        initial_nodes=None, 
        scene_info={}, 
        modes=['default', 'random'], 
        n_samples=1,
        intervene={},
        rng=None,
        render_func=None,
        **render_func_args):
    '''
    Renders scenes via blender with desired corruptions based on the model.

    Parameters:
        model: Instance of CausalModel class or WrapperModel class. Specifies the type of corruptions to apply to the scene.
        initial_nodes: Nodes in the scene's node_tree object to keep during each render call.
        scene_info: Dictionary storing meta information about the scene
        modes: List of types of corruptions to render the scene with
            default: Render the scene without any corruptions
            random: Render the scene with random corruptions
        n_samples: The number of times to generate a scene with random corruptions for each model (only used if 'random' is a provided mode)
        intervene: Dictionary storing intervening values to render scenes with
        rng: Random Number Generator (numpy) for reproducability of corruptions
        render_func: A callable function which is what will render the blender scene. 
                     If not provided, the default blender render scene will be used.
        render_func_args: Arguments to be passed to "render_func" during call.
    '''
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    render_args = bpy.context.scene.render

    # default render function call and arguments
    if render_func is None:
        render_func = bpy.ops.render.render
        render_func_args = {'write_still': True}

    # creating copy to not modify original modes instance
    if isinstance(modes, str):
        modes_arr = [modes]
    else:
        modes_arr = [mode for mode in modes]

    # get initial nodes of the tree
    if initial_nodes is None:
        initial_nodes = []
        for node in tree.nodes:
            initial_nodes.append(node.name)

    # first render default image with no corruptions
    # needs to be done before any type of corruption sampling/blender settings are updated
    # i.e. default needs to be run before other modes
    og_path = render_args.filepath
    ret = None
    if 'default' in modes_arr:
        try:
            ret = render_func(**render_func_args)
            logging.info('Finished rendering default scene')
        except Exception as e:
            logging.info('Error in rendering default scene')
            logging.info(e)
            ret = None

        # did render_func change the rendering filepath?
        if render_args.filepath != og_path:
            og_path = '/'.join(render_args.filepath.split('/'))

        # clear nodes created during rendering process (i.e. output image nodes)
        reset_tree(tree, initial_nodes)
        while 'default' in modes_arr:
            modes_arr.remove('default')

    # ensure model is wrapped
    if not isinstance(model, WrapperModel):
        model = WrapperModel(model.__class__, None, dags=model)

    # render with randomly corrupted scene 
    if 'random' in modes_arr:
        ret = render_random_corrupted_scene(model,
                render_args, 
                tree, 
                initial_nodes, 
                og_path, 
                scene_info, 
                n_samples, 
                intervene,
                rng,
                render_func, 
                render_func_args)

    return ret


def render_random_corrupted_scene(model, 
        render_args, 
        tree, 
        initial_nodes, 
        og_path, 
        scene_info, 
        n_samples,
        intervene,
        rng,
        render_func,
        render_func_args):
    '''
    Renders a corrupted blender scenes with random corruption levels. A total of n_samples scenes are rendered for each model.

    Parameters:
        model: WrapperModel instance which contains a list of CausalModels to sample corruptions from
        render_args: The blender render settings
        tree: The blender scene's active context tree
        initial_nodes: The initial nodes populating the blender tree before sampling corruptions
        og_path: The filepath of the non-corrupted png of the blender scene
        scene_info: Dictionary storing information about the scene and its objects
        n_samples: The number of times to randomly sample corruption values from each CausalModel 
        intervene: Dictionary storing intervening values to render scenes with
        rng: Random Number Generator (numpy) for reproducability of corruptions
        render_func: A callable function which is what will render the blender scene. 
        render_func_args: Dictionary of arguments to be passed to "render_func" during call.
    '''
    render_args.filepath = og_path.replace('.png', '') + '_random.png'
    render_args.filepath = remove_leading_fpath_underscore(render_args.filepath)

    scene_info['corruptions'] = {}

    for n in range(model.n_dags):
        scene_info['corruptions'][n] = {}

        for q in range(n_samples):
            # get random corruption values from the nth model
            img_params = model.sample(scene=scene_info, idx=n, intervene=intervene, rng=rng)
            fn_base = og_path.replace('.png', '') + '_%03d_%03d.png' % (n,q)
            fn_base = remove_leading_fpath_underscore(fn_base)
            root = tree.nodes['Render Layers'].outputs['Image']

            reset_tree(tree, initial_nodes)

            # update the context tree to use the sampled corruption values
            model.update_tree(root, tree, params=img_params, fn_base=fn_base, idx=n)

            _img_corruptions = []
            for name, v in img_params.items():
                info = {'node': name, 'params': {}}
                for param, param_val in v.items():
                    info['params'][param] = param_val
                _img_corruptions.append(info)
            scene_info['corruptions'][n][q] = _img_corruptions

            try:
                ret = render_func(**render_func_args)
                logging.info('Finished rendering randomly corrupt scene')
            except Exception as e:
                logging.info('Error in rendering randomly corrupt scene')
                logging.info(e)
                ret = None

    return ret


def remove_leading_fpath_underscore(fpath):
    '''
        Removes leading underscores from a filepath.
    '''
    stem = fpath.split('/')[-1]
    base = '/'.join(fpath.split('/')[:-1])
    if stem[0] == '_':
        stem = stem[1:]
    fpath = f'{base}/{stem}'
    return fpath
