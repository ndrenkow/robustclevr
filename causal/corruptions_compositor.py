# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

# Code adapted from: https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py

import os
import bpy
from .registry import Factory

__all__ = [
    'output_file',
    'defocus',
    'gamma',
    'blur',
    'glare'
]


@Factory.register('output_file')
def output_file(tree, output_image, src_node_name, **kwargs):
    out = tree.nodes.new('CompositorNodeOutputFile')
    out.base_path = os.path.dirname(output_image)
    out.file_slots[0].path = os.path.basename(output_image).replace('.png', '-{}-'.format(src_node_name))
    return out


@Factory.register('defocus')
def defocus(tree, z=1., f_stop=128, **kwargs):
    node = tree.nodes.new(type='CompositorNodeDefocus')
    node.inputs[1].default_value = z
    node.f_stop = f_stop
    return node


@Factory.register('gamma')
def gamma(tree, gamma=1.0, **kwargs):
    node = tree.nodes.new(type='CompositorNodeGamma')
    node.inputs[1].default_value = gamma
    return node


@Factory.register('blur')
def blur(tree, size_x=1, size_y=1, **kwargs):
    node = tree.nodes.new(type='CompositorNodeBlur')
    node.size_x = int(size_x)
    node.size_y = int(size_y)
    return node


@Factory.register('glare')
def glare(tree, glare_type='STREAKS', mix=-1, **kwargs):
    node = tree.nodes.new(type='CompositorNodeGlare')
    node.glare_type = glare_type
    node.angle_offset = kwargs.get('angle_offset', 0)
    node.streaks = kwargs.get('streaks', 1)
    node.size = kwargs.get('size', 0)
    node.iterations = kwargs.get('iterations', 3)
    node.mix = mix
    node.color_modulation = kwargs.get('color_modulation', 0.)
    return node


@Factory.register('pixelate')
def pixelate(tree, downscale=0.2, upscale=5, **kwargs):
    if downscale == 1 and upscale == 1:
        # Do add 0 as a no-op
        node = tree.nodes.new(type='CompositorNodeMath')
        node.operation = 'ADD'
        node.inputs[1].default_values = 0
        return node
    
    down = tree.nodes.new(type='CompositorNodeScale')
    down.inputs[1].default_value = downscale  # X
    down.inputs[2].default_value = downscale  # Y
    
    up = tree.nodes.new(type='CompositorNodeScale')
    up.inputs[1].default_value = upscale  # X
    up.inputs[2].default_value = upscale  # Y
    
    node = tree.nodes.new(type='CompositorNodePixelate')
    tree.links.new(down.outputs['Image'], node.inputs['Image'])
    tree.links.new(node.inputs['Image'], up.outputs['Image'])
    return [down, node, up]


@Factory.register('lensdist')
def lensdist(tree, distort, dispersion, projector=False, jitter=False, fit=False, **kwargs):
    node = tree.nodes.new(type='CompositorNodeLensdist')
    node.inputs[1].default_value = distort
    node.inputs[2].default_value = dispersion
    node.use_projector = projector
    node.use_jitter = jitter
    node.use_fit = fit
    return node


@Factory.register('bright_contrast')
def bright_contrast(tree, bright, contrast, **kwargs):
    node = tree.nodes.new(type='CompositorNodeBrightContrast')
    node.inputs[1].default_value = bright
    node.inputs[2].default_value = contrast
    return node


@Factory.register('directional_blur')
def directional_blur(tree, iterations=0, wrap=False, center_x=0.5, center_y=0.5, distance=0., angle=0., spin=0.,
                     zoom=0., **kwargs):
    node = tree.nodes.new(type='CompositorNodeDBlur')
    node.iterations = iterations
    node.use_wrap = wrap
    node.center_x = center_x
    node.center_y = center_y
    node.distance = distance
    node.angle = angle
    node.spin = spin
    node.zoom = zoom
    return node


@Factory.register('displace')
def displace(tree, scale_x=0, scale_y=0, **kwargs):
    node = tree.nodes.new(type='CompositorNodeDisplace')
    node.inputs[2].default_value = scale_x
    node.inputs[3].default_value = scale_y
    return node


@Factory.register('vector_blur')
def vector_blur(tree, factor=1, samples=16, speed_min=0, speed_max=1, z=0, **kwargs):
    """ Currently work in progress """
    node = tree.nodes.new(type='CompositorNodeVecBlur')
    node.inputs[1].default_value = z
    node.factor = factor
    node.samples = samples
    node.speed_min = speed_min
    node.speed_max = speed_max
    return node


@Factory.register('noise_shader')
def noise_shader(tree, absorption=0.1, density=0.08, **kwargs):
    """ tree should be scene.world.node_tree """
    a = tree.nodes.new('ShaderNodeVolumeAbsorption')
    a.inputs[1].default_value = absorption
    v = tree.nodes.new('ShaderNodeVolumeScatter')
    v.inputs[1].default_value = density
    m = tree.nodes.new('ShaderNodeAddShader')
    tree.links.new(a.outputs[0], m.inputs[0])
    tree.links.new(v.outputs[0], m.inputs[1])
    return m


@Factory.register('noise')
def noise(tree, factor=0., blend_type='ADD', **kwargs):
    tex = tree.nodes.new(type='CompositorNodeTexture')
    tex.texture = bpy.data.textures.new('Noise', 'NOISE')

    mix = tree.nodes.new(type='CompositorNodeMixRGB')
    mix.blend_type = blend_type
    mix.inputs[0].default_value = factor

    tree.links.new(tex.outputs[1], mix.inputs[2])
    return mix


@Factory.register('clouds')
def clouds(tree, factor=0., blend_type='ADD', **kwargs):
    tex = tree.nodes.new(type='CompositorNodeTexture')
    tex.texture = bpy.data.textures.new('Cloud', 'CLOUDS')

    mix = tree.nodes.new(type='CompositorNodeMixRGB')
    mix.blend_type = blend_type
    mix.inputs[0].default_value = factor

    tree.links.new(tex.outputs[1], mix.inputs[2])
    return mix


@Factory.register('rain')
def rain(tree, factor=0.1, strength=0.7, angle=0., **kwargs):
    tex = tree.nodes.new(type='CompositorNodeTexture')
    tex.texture = bpy.data.textures.new('Cloud', 'CLOUDS')
    tex.inputs[1].default_value = (30, 0.5, 1)
    
    rotate = tree.nodes.new(type='CompositorNodeRotate')
    rotate.filter_type = 'BILINEAR'
    rotate.inputs[1].default_value = angle
    
    scale = tree.nodes.new(type='CompositorNodeScale')
    scale.space = 'RELATIVE'
    scale.inputs[1].default_value = 2.
    scale.inputs[2].default_value = 2.
    
    ramp = tree.nodes.new(type='CompositorNodeValToRGB')
    ramp.color_ramp.evaluate(strength)

    multiply = tree.nodes.new(type='CompositorNodeMath')
    multiply.operation = 'MULTIPLY'
    multiply.inputs[1].default_value = 1.

    mix = tree.nodes.new(type='CompositorNodeMixRGB')
    mix.blend_type = 'ADD'
    mix.inputs[0].default_value = factor

    tree.links.new(tex.outputs[0], rotate.inputs[0])
    tree.links.new(rotate.outputs[0], scale.inputs[0])
    tree.links.new(ramp.inputs[0], scale.outputs[0])
    tree.links.new(ramp.outputs[0], multiply.inputs[0])
    tree.links.new(multiply.outputs[0], mix.inputs[2])
    return mix

@Factory.register('snow')
def rain(tree, factor=0.1, strength=0.7, angle=-10., **kwargs):
    tex = tree.nodes.new(type='CompositorNodeTexture')
    tex.texture = bpy.data.textures.new('Cloud', 'CLOUDS')
    tex.inputs[1].default_value = (0.5, 0.5, 1)
    
    rotate = tree.nodes.new(type='CompositorNodeRotate')
    rotate.filter_type = 'BILINEAR'
    rotate.inputs[1].default_value = angle
    
    scale = tree.nodes.new(type='CompositorNodeScale')
    scale.space = 'RELATIVE'
    scale.inputs[1].default_value = 2.
    scale.inputs[2].default_value = 2.
    
    ramp = tree.nodes.new(type='CompositorNodeValToRGB')
    ramp.color_ramp.evaluate(strength)

    multiply = tree.nodes.new(type='CompositorNodeMath')
    multiply.operation = 'MULTIPLY'
    multiply.inputs[1].default_value = 1.

    mix = tree.nodes.new(type='CompositorNodeMixRGB')
    mix.blend_type = 'ADD'
    mix.inputs[0].default_value = factor

    tree.links.new(tex.outputs[0], rotate.inputs[0])
    tree.links.new(rotate.outputs[0], scale.inputs[0])
    tree.links.new(ramp.inputs[0], scale.outputs[0])
    tree.links.new(ramp.outputs[0], multiply.inputs[0])
    tree.links.new(multiply.outputs[0], mix.inputs[2])
    return mix
