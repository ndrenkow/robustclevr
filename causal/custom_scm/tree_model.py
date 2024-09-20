# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import numpy as np
import torch
from ..dist_utils import DiscreteUniform, Categorical, HalfNormal, Normal
from ..causal_model import ModelBase
from ..registry import Factory


@Factory.register('TreeModel')
class TreeModel(ModelBase):
    def __init__(self):
        self.node_list = [
            {"name": "BC", "causal_func": "bright_contrast"},
            {"name": "G", "causal_func": "gamma"},
            {"name": "D", "causal_func": "defocus"},
            {"name": "B", "causal_func": "blur"},
            {"name": "L", "causal_func": "lensdist"},
            {"name": "DB", "causal_func": "directional_blur"},
            {"name": "DS", "causal_func": "displace"},
            {"name": "N", "causal_func": "noise"},
            {"name": "C", "causal_func": "clouds"},
            {"name": "GL", "causal_func": "glare"}
        ]
        self.edge_list = [["root", "L"], ["root", "DB"], ["root", "G"], ["root", "BC"],
                 ["root", "D"], ["root", "B"], ["root", "N"], ["root", "C"],
                 ["root", "GL"]]

        for node in self.node_list:
            node["corruption_func"] = node["causal_func"]
            node["causal_func"] = getattr(self, node["causal_func"])
            assert node["causal_func"] is not None

        super().__init__()

    def gamma(self, scene=None, **kwargs):
        n_bins = 100
        offset = 1
        discrete_unif = DiscreteUniform(n_bins)
        gamma, logprob = discrete_unif.sample()
        gamma *= 2.
        gamma += offset  # gamma \in [1, 3)
        return dict(gamma=gamma, logprob=dict(gamma=logprob))
    
    def blur(self, scene=None, **kwargs):
        max_kernel_size = 11
        cat = Categorical(n_bins=max_kernel_size)
        sz, logprob = cat.sample()
        # sz *= max_kernel_size
        sz += 1.  # sz \in {1,...,max_kernel_size}
        return dict(size_x=sz, size_y=sz,
                    logprob=dict(size_x=logprob, size_y=logprob))
    
    def defocus(self, scene=None, **kwargs):
        cat = Categorical(n_bins=10)
        z, z_logprob = cat.sample()
        z += 1.  # z \in {1, 10}
        cat2 = Categorical(n_bins=64)
        f_stop, f_stop_logprob = cat2.sample()
        f_stop += 64  # f_stop \in {64,...,128}
        return dict(z=z, f_stop=f_stop,
                    logprob=dict(z=z_logprob, f_stop=f_stop_logprob))
    
    def pixelate(self, scene=None, **kwargs):
        unif = DiscreteUniform(n_bins=100)
        do_pixelate, logprob = unif.sample()
        do_pixelate = do_pixelate > 0.5
        return dict(downscale=0.2 if do_pixelate else 1, upscale=5 if do_pixelate else 1,
                    logprob=dict(downscale=logprob, upscale=logprob))
    
    def lensdist(self, scene=None, **kwargs):
        unif = DiscreteUniform(n_bins=100)
        distort, distort_logprob = unif.sample()
        distort *= 0.1  # distort \in [0, 0.1)
        fit, fit_logprob = True, 0.
        dispersion, disp_logprob = unif.sample()
        dispersion *= 0.5  # \in [0, 0.5)
        return dict(distort=distort, dispersion=dispersion, projector=False, jitter=False, fit=fit,
                    logprob=dict(distort=distort_logprob, fit=fit_logprob, dispersion=disp_logprob))
    
    def directional_blur(self, scene=None, **kwargs):
        unif = DiscreteUniform(n_bins=100)
        distance, distance_logprob = unif.sample()
        distance *= 0.1  # \in [0, 0.05)
        angle, angle_logprob = unif.sample()
        angle *= 2 * np.pi  # \in [0, 2*pi)
        return dict(iterations=12, wrap=True, center_x=0.5, center_y=0.5, distance=distance,
                    angle=angle, spin=0., zoom=0.,
                    logprob=dict(distance=distance_logprob, angle=angle_logprob))
    
    def displace(self, scene=None, **kwargs):
        unif = DiscreteUniform(n_bins=100)
        scale, logprob = unif.sample()
        scale *= 100.  # \in [0, 100)
        return dict(scale_x=scale, scale_y=scale, logprob=dict(scale_x=logprob, scale_y=logprob))
    
    def noise(self, scene=None, **kwargs):
        unif = DiscreteUniform(n_bins=100)
        factor, logprob = unif.sample()
        factor *= 0.25  # \in [0, 0.25)
        return dict(factor=factor, logprob=dict(factor=logprob))
    
    def clouds(self, scene=None, **kwargs):
        unif = DiscreteUniform(n_bins=100)
        factor, logprob = unif.sample()
        factor *= 0.30  # \in [0, 0.3)
        return dict(factor=factor, logprob=dict(factor=logprob))
    
    def bright_contrast(self, scene=None, **kwargs):
        unif = DiscreteUniform(n_bins=100)
        bright, bright_logprob = unif.sample()
        contrast, contrast_logprob = unif.sample()
        bright = 4 * bright - 20  # \in (-20, -16]
        contrast = 20 * contrast - 100  # \in (-100, -80]
        return dict(bright=bright, contrast=contrast, logprob=dict(bright=bright_logprob, contrast=contrast_logprob))
    
    def glare(self, scene=None, **kwargs):
        glare_type = 'SIMPLE_STAR'
        unif = DiscreteUniform(n_bins=100)
        mix, logprob = unif.sample()
        mix -= 0.5
        mix *= 1.  # \in [-0.5, 0.5)
        angle_offset, angle_logprob = unif.sample()
        angle_offset *= 2 * np.pi  # \in [0, 2*pi)
        return dict(glare_type=glare_type, mix=mix, angle_offset=angle_offset,
                    logprob=dict(mix=logprob, angle_offset=angle_logprob))


@Factory.register('TreeModelV2')
class TreeModelV2(ModelBase):
    def __init__(self):
        self.node_list = [
            {"name": "BC", "causal_func": "bright_contrast"},
            {"name": "G", "causal_func": "gamma"},
            {"name": "D", "causal_func": "defocus"},
            {"name": "B", "causal_func": "blur"},
            {"name": "L", "causal_func": "lensdist"},
            {"name": "DB", "causal_func": "directional_blur"},
            {"name": "DS", "causal_func": "displace"},
            {"name": "N", "causal_func": "noise"},
            {"name": "C", "causal_func": "clouds"},
            {"name": "GL", "causal_func": "glare"}
        ]
        self.edge_list = [["root", "L"], ["root", "DB"], ["root", "G"], ["root", "BC"],
                 ["root", "D"], ["root", "B"], ["root", "N"], ["root", "C"],
                 ["root", "GL"]]

        for node in self.node_list:
            node["corruption_func"] = node["causal_func"]
            node["causal_func"] = getattr(self, node["causal_func"])
            assert node["causal_func"] is not None

        super().__init__()
    
    def gamma(self, scene=None, **kwargs):
        n_bins = 100
        offset = 1
        discrete_hn = HalfNormal(scale=1)
        gamma, logprob = discrete_hn.sample()
        gamma += offset  # gamma \in [1, inf)
        return dict(gamma=gamma, logprob=dict(gamma=logprob))
    
    def blur(self, scene=None, **kwargs):
        max_kernel_size = 11
        logits = torch.flip(torch.arange(max_kernel_size) + 1, dims=[0])  # Lower sizes have higher weight
        cat = Categorical(logits=logits.float())
        sz, logprob = cat.sample()
        sz += 1.  # sz \in {1,...,max_kernel_size}
        return dict(size_x=sz, size_y=sz,
                    logprob=dict(size_x=logprob, size_y=logprob))
    
    def defocus(self, scene=None, **kwargs):
        hn = HalfNormal(scale=3)
        z, z_logprob = hn.sample()
        z += 1.  # z ~ N(1, 3)
        cat2 = Categorical(n_bins=64)
        f_stop, f_stop_logprob = cat2.sample()
        f_stop += 64  # f_stop \in {64,...,128}
        return dict(z=z, f_stop=f_stop,
                    logprob=dict(z=z_logprob, f_stop=f_stop_logprob))
    
    def lensdist(self, scene=None, **kwargs):
        hn = HalfNormal(scale=1.)
        distort, distort_logprob = hn.sample()
        distort *= 0.1  # distort \in [0, 0.1)
        fit, fit_logprob = True, 0.
        dispersion, disp_logprob = hn.sample()
        dispersion *= 0.5  # \in [0, 0.5)
        return dict(distort=distort, dispersion=dispersion, projector=False, jitter=False, fit=fit,
                    logprob=dict(distort=distort_logprob, fit=fit_logprob, dispersion=disp_logprob))
    
    def directional_blur(self, scene=None, **kwargs):
        hn = HalfNormal(scale=0.1)
        distance, distance_logprob = hn.sample()
        unif = DiscreteUniform(n_bins=100)
        angle, angle_logprob = unif.sample()
        angle *= 2 * np.pi  # \in [0, 2*pi)
        return dict(iterations=12, wrap=True, center_x=0.5, center_y=0.5, distance=distance,
                    angle=angle, spin=0., zoom=0.,
                    logprob=dict(distance=distance_logprob, angle=angle_logprob))
    
    def displace(self, scene=None, **kwargs):
        hn = HalfNormal(scale=1)
        scale, logprob = hn.sample()
        scale *= 100.  # ~ N(0, 100)
        return dict(scale_x=scale, scale_y=scale, logprob=dict(scale_x=logprob, scale_y=logprob))
    
    def noise(self, scene=None, **kwargs):
        hn = HalfNormal(scale=0.25)
        factor, logprob = hn.sample()
        return dict(factor=factor, logprob=dict(factor=logprob))
    
    def clouds(self, scene=None, **kwargs):
        hn = HalfNormal(scale=0.3)
        factor, logprob = hn.sample()
        return dict(factor=factor, logprob=dict(factor=logprob))
    
    def bright_contrast(self, scene=None, **kwargs):
        normal = Normal(scale=1)
        bright, bright_logprob = normal.sample()
        bright = -18 + 2 * bright  # ~ N(-18, 2)
        contrast, contrast_logprob = normal.sample()
        contrast = -90 + 10 * contrast  # ~ N(-90, 10)
        return dict(bright=bright, contrast=contrast, logprob=dict(bright=bright_logprob, contrast=contrast_logprob))
    
    def glare(self, scene=None, **kwargs):
        glare_type = 'SIMPLE_STAR'
        normal = Normal(loc=0, scale=0.5)
        mix, logprob = normal.sample()
        unif = DiscreteUniform(n_bins=100)
        angle_offset, angle_logprob = unif.sample()
        angle_offset *= 2 * np.pi  # \in [0, 2*pi)
        return dict(glare_type=glare_type, mix=mix, angle_offset=angle_offset,
                    logprob=dict(mix=logprob, angle_offset=angle_logprob))


@Factory.register('TreeModelV3')
class TreeModelV3(ModelBase):
    def __init__(self):
        self.node_list = [
            {"name": "BC", "causal_func": "bright_contrast"},
            {"name": "G", "causal_func": "gamma"},
            {"name": "D", "causal_func": "defocus"},
            {"name": "B", "causal_func": "blur"},
            {"name": "L", "causal_func": "lensdist"},
            {"name": "DB", "causal_func": "directional_blur"},
            {"name": "DS", "causal_func": "displace"},
            {"name": "N", "causal_func": "noise"},
            {"name": "C", "causal_func": "clouds"},
            {"name": "GL", "causal_func": "glare"}
        ]
        self.edge_list = [["root", "L"], ["root", "DB"], ["root", "G"], ["root", "BC"],
                 ["root", "D"], ["root", "B"], ["root", "N"], ["root", "C"],
                 ["root", "GL"]]

        for node in self.node_list:
            node["corruption_func"] = node["causal_func"]
            node["causal_func"] = getattr(self, node["causal_func"])
            assert node["causal_func"] is not None

        super().__init__()
    
    def gamma(self, scene=None, **kwargs):
        n_bins = 100
        offset = 1
        discrete_hn = HalfNormal(scale=1)
        gamma, logprob = discrete_hn.sample()
        gamma += offset  # gamma \in [1, inf)
        return dict(gamma=gamma, logprob=dict(gamma=logprob))
    
    def blur(self, scene=None, **kwargs):
        max_kernel_size = 11
        hn = HalfNormal(scale=1)
        sz, logprob = hn.sample()
        sz *= max_kernel_size
        sz = int(sz)
        # logits = torch.flip(torch.arange(max_kernel_size) + 1, dims=[0])  # Lower sizes have higher weight
        # cat = Categorical(logits=logits.float())
        # sz, logprob = cat.sample()
        sz += 1.  # sz \in {1,...,max_kernel_size}
        return dict(size_x=sz, size_y=sz,
                    logprob=dict(size_x=logprob, size_y=logprob))
    
    def defocus(self, scene=None, **kwargs):
        hn = HalfNormal(scale=3)
        z, z_logprob = hn.sample()
        z += 1.  # z ~ N(1, 3)
        cat2 = Categorical(n_bins=64)
        f_stop, f_stop_logprob = cat2.sample()
        f_stop += 64  # f_stop \in {64,...,128}
        return dict(z=z, f_stop=f_stop,
                    logprob=dict(z=z_logprob, f_stop=f_stop_logprob))
    
    def lensdist(self, scene=None, **kwargs):
        hn = HalfNormal(scale=1.)
        distort, distort_logprob = hn.sample()
        distort *= 0.2  # distort \in [0, 0.2)
        fit, fit_logprob = True, 0.
        dispersion, disp_logprob = hn.sample()
        dispersion *= 0.7  # \in [0, 0.7)
        return dict(distort=distort, dispersion=dispersion, projector=False, jitter=False, fit=fit,
                    logprob=dict(distort=distort_logprob, fit=fit_logprob, dispersion=disp_logprob))
    
    def directional_blur(self, scene=None, **kwargs):
        hn = HalfNormal(scale=0.2)
        distance, distance_logprob = hn.sample()
        unif = DiscreteUniform(n_bins=100)
        angle, angle_logprob = unif.sample()
        angle *= 2 * np.pi  # \in [0, 2*pi)
        return dict(iterations=12, wrap=True, center_x=0.5, center_y=0.5, distance=distance,
                    angle=angle, spin=0., zoom=0.,
                    logprob=dict(distance=distance_logprob, angle=angle_logprob))
    
    def displace(self, scene=None, **kwargs):
        hn = HalfNormal(scale=1.5)
        scale, logprob = hn.sample()
        scale *= 100.  # ~ N(0, 100)
        return dict(scale_x=scale, scale_y=scale, logprob=dict(scale_x=logprob, scale_y=logprob))
    
    def noise(self, scene=None, **kwargs):
        hn = HalfNormal(scale=0.25)
        factor, logprob = hn.sample()
        return dict(factor=factor, logprob=dict(factor=logprob))
    
    def clouds(self, scene=None, **kwargs):
        hn = HalfNormal(scale=0.3)
        factor, logprob = hn.sample()
        return dict(factor=factor, logprob=dict(factor=logprob))
    
    def bright_contrast(self, scene=None, **kwargs):
        normal = Normal(scale=1)
        bright, bright_logprob = normal.sample()
        bright = -18 + 2 * bright  # ~ N(-18, 2)
        contrast, contrast_logprob = normal.sample()
        contrast = -90 + 10 * contrast  # ~ N(-90, 10)
        return dict(bright=bright, contrast=contrast, logprob=dict(bright=bright_logprob, contrast=contrast_logprob))
    
    def glare(self, scene=None, **kwargs):
        glare_type = 'SIMPLE_STAR'
        normal = Normal(loc=0, scale=0.5)
        mix, logprob = normal.sample()
        unif = DiscreteUniform(n_bins=100)
        angle_offset, angle_logprob = unif.sample()
        angle_offset *= 2 * np.pi  # \in [0, 2*pi)
        return dict(glare_type=glare_type, mix=mix, angle_offset=angle_offset,
                    logprob=dict(mix=logprob, angle_offset=angle_logprob))
