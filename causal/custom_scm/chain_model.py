# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import numpy as np
from ..dist_utils import DiscreteUniform, Categorical, HalfNormal
from ..causal_model import ModelBase
from ..registry import Factory


def get_parents(key, edge_list):
    parents = []
    for edge in edge_list:
        if edge[1] == key and edge[0] != "root":
            parents.append(edge[0])
    return parents

def get_values(values, keys, node, edge_list):
    requested = []
    for key in keys:
        parents = get_parents(node, edge_list)
        for parent in parents:
            if key in values[parent]:
                requested.append(values[parent][key])
                break
    assert len(requested) == len(keys)
    return tuple(requested)



@Factory.register('ChainModel')
class ChainModel(ModelBase):
    def __init__(self):
        self.node_list = [
            {"name": "C", "causal_func": "clouds"},
            {"name": "G", "causal_func": "gamma"},
            {"name": "B", "causal_func": "blur"},
            {"name": "L", "causal_func": "lensdist"},
            {"name": "DB", "causal_func": "directional_blur"},
            {"name": "D", "causal_func": "defocus"},
            {"name": "N", "causal_func": "noise"}
        ]
        self.alt = [["root", "G"], ["G", "C"], ["C", "DB"], ["DB", "L"], ["L", "N"]],
        self.edge_list = [["root", "C"], ["C", "B"], ["B", "G"], ["G", "L"], ["L", "DB"], ["DB", "D"], ["D", "N"]]

        for node in self.node_list:
            node["corruption_func"] = node["causal_func"]
            node["causal_func"] = getattr(self, node["causal_func"])
            assert node["causal_func"] is not None

        super().__init__()
    
    def clouds(self, scene=None, **kwargs):
        d = DiscreteUniform(n_bins=100)
        v, lp = d.sample()
        p_cloud = 0.75
        if v > p_cloud:
            scale = 0.3
            hn = HalfNormal(scale=scale)
            val, logprob = hn.sample()
            logprob += np.log(p_cloud)
        else:
            val = 0.
            logprob = np.log(0.25)
        return dict(factor=val, logprob=dict(factor=logprob))
    
    def blur(self, scene=None, **kwargs):
        factor, = get_values(kwargs, ['factor'], "B", self.edge_list)
        if factor > 0.2:
            sz = 1
            logprob = 0.
        else:
            max_kernel_size = 9
            cat = Categorical(n_bins=max_kernel_size)
            sz, logprob = cat.sample()
            sz += 1
        return dict(size_x=sz, size_y=sz,
                    logprob=dict(size_x=logprob, size_y=logprob))
    
    def gamma(self, scene=None, **kwargs):
        size_x, size_y = get_values(kwargs, ['size_x', 'size_y'], "G", self.edge_list)
        d = DiscreteUniform(n_bins=100)
        val, logprob = d.sample()
        if size_x <= 3:
            gamma = 0.1 * val
        elif size_x > 3:
            gamma = val
        else:
            gamma = 0.
            logprob = 0
        gamma += 1.
        return dict(gamma=gamma, logprob=dict(gamma=logprob))
    
    def lensdist(self, scene=None, **kwargs):
        gamma, = get_values(kwargs, ['gamma'], "L", self.edge_list)
        d = DiscreteUniform(n_bins=100)
        val, logprob = d.sample()
        if gamma > 1.2:
            distort = 0.05 * val
            p_distort = logprob
            dispersion = 0
            p_dispersion = 0.
        elif gamma > 1.0:
            dispersion = 0.5 * val
            p_dispersion = logprob
            distort = 0.
            p_distort = 0.
        else:
            distort = 0
            p_distort = 0.
            dispersion = 0.
            p_dispersion = 0.
        fit = distort > 0
        return dict(distort=distort, dispersion=dispersion, projector=False, jitter=False, fit=fit,
                    logprob=dict(distort=p_distort, dispersion=p_dispersion))
    
    def directional_blur(self, scene=None, **kwargs):
        distort, = get_values(kwargs, ['distort'], "DB", self.edge_list)
        d = DiscreteUniform(n_bins=100)
        val, logprob = d.sample()
        val2, logprob2 = d.sample()
        if distort == 0:
            zoom = 0.1 * val
            distance = 0.05 * val2
        else:
            zoom = 0
            logprob = 0.
            distance = 0
        return dict(iterations=12, wrap=False, center_x=0.5, center_y=0.5, distance=distance, angle=0, spin=0,
                    zoom=zoom,
                    logprob=dict(zoom=logprob, distance=logprob2))
    
    def defocus(self, scene=None, **kwargs):
        zoom, distance = get_values(kwargs, ['zoom', 'distance'], "D", self.edge_list)
        if zoom == 0 or distance == 0:
            cat = Categorical(n_bins=10)
            z, z_logprob = cat.sample()
            z += 1.  # z \in {1, 10}
            cat2 = Categorical(n_bins=64)
            f_stop, f_stop_logprob = cat2.sample()
            f_stop += 64  # f_stop \in {64,...,128}
        else:
            z = 1
            f_stop = 128
            z_logprob = 0.
            f_stop_logprob = 0.
        return dict(z=z, f_stop=f_stop,
                    logprob=dict(z=z_logprob, f_stop=f_stop_logprob))
    
    def noise(self, scene=None, **kwargs):
        z, f_stop = get_values(kwargs, ['z', 'f_stop'], "N", self.edge_list)
        d = DiscreteUniform(n_bins=100)
        val, logprob = d.sample()
        if z < 4 or f_stop > 100:
            factor = 0.2 * val
        else:
            factor = 0.05 * val
        return dict(factor=factor, logprob=dict(factor=logprob))


@Factory.register('ChainModelV2')
class ChainModelV2(ModelBase):
    def __init__(self):
        self.node_list = [
            {"name": "C", "causal_func": "clouds"},
            {"name": "G", "causal_func": "gamma"},
            {"name": "B", "causal_func": "blur"},
            {"name": "L", "causal_func": "lensdist"},
            {"name": "DB", "causal_func": "directional_blur"},
            {"name": "D", "causal_func": "defocus"},
            {"name": "N", "causal_func": "noise"}
        ]
        self.edge_list = [["root", "G"], ["G", "L"], ["L", "D"], ["D", "DB"], ["DB", "B"], ["B", "C"], ["C", "N"]]

        for node in self.node_list:
            node["corruption_func"] = node["causal_func"]
            node["causal_func"] = getattr(self, node["causal_func"])
            assert node["causal_func"] is not None

        super().__init__()
    
    def gamma(self, scene=None, **kwargs):
        d = DiscreteUniform(n_bins=100)
        gamma, logprob = d.sample()
        gamma += 1.
        return dict(gamma=gamma, logprob=dict(gamma=logprob))
    
    def lensdist(self, scene=None, **kwargs):
        gamma, = get_values(kwargs, ['gamma'], "L", self.edge_list)
        d = DiscreteUniform(n_bins=100)
        distort, logprob = d.sample()
        dispersion, logprob2 = d.sample()
        distort = 0.
        dispersion *= 0.5  # \in [0, 0.5)
        scale = 1.5
        if 1.8 < gamma < 2:
            distort *= scale
            dispersion *= scale
        fit = distort > 0
        return dict(distort=distort, dispersion=dispersion, projector=False, jitter=False, fit=fit,
                    logprob=dict(distort=logprob, dispersion=logprob2))
    
    def defocus(self, scene=None, **kwargs):
        distort, dispersion = get_values(kwargs, ['distort', 'dispersion'], "D", self.edge_list)
        d = DiscreteUniform(n_bins=100)
        z, logprob = d.sample()
        f_stop, logprob2 = d.sample()
        if 0.01 > distort > 0.015 or 0.5 < dispersion < 0.75:
            z = z * 5 + 1
            f_stop = f_stop * 32 + 96
        else:
            z = z * 10 + 1
            f_stop = f_stop * 64 + 64
        return dict(z=z, f_stop=f_stop,
                    logprob=dict(z=logprob, f_stop=logprob2))
    
    def directional_blur(self, scene=None, **kwargs):
        z, f_stop = get_values(kwargs, ['z', 'f_stop'], "DB", self.edge_list)
        d = DiscreteUniform(n_bins=100)
        distance, logprob = d.sample()
        zoom, logprob2 = d.sample()
        
        if z > 5 or f_stop > 96:
            distance *= 0.02
            zoom *= 0.02
        else:
            distance *= 0.1
            zoom *= 0.1
        return dict(iterations=12, wrap=False, center_x=0.5, center_y=0.5, distance=distance, angle=0, spin=0,
                    zoom=zoom, logprob=dict(zoom=logprob, distance=logprob2))
    
    def blur(self, scene=None, **kwargs):
        zoom, distance = get_values(kwargs, ['zoom', 'distance'], "B", self.edge_list)
        max_kernel_size = 11
        cat = Categorical(n_bins=max_kernel_size)
        sz1, logprob = cat.sample()
        cat2 = Categorical(n_bins=max_kernel_size // 2)
        sz2, logprob2 = cat2.sample()
        
        if zoom > 0.05 or distance > 0.05:
            sz = sz2
        else:
            sz = sz1
            
        return dict(size_x=sz, size_y=sz,
                    logprob=dict(size_x=logprob, size_y=logprob))
    
    def clouds(self, scene=None, **kwargs):
        size_x, size_y = get_values(kwargs, ['size_x', 'size_y'], "C", self.edge_list)
        d = DiscreteUniform(n_bins=100)
        val, logprob = d.sample()
        if 3 <= size_x < 7:
            val *= 0.1
        elif size_x < 3:
            val *= 0.2
        else:
            val *= 0.
        return dict(factor=val, logprob=dict(factor=logprob))
    
    def noise(self, scene=None, **kwargs):
        factor, = get_values(kwargs, ['factor'], "N", self.edge_list)
        d = DiscreteUniform(n_bins=100)
        val, logprob = d.sample()
        if factor > 0.1:
            val *= 0.01
        else:
            val *= 0.2
        return dict(factor=val, logprob=dict(factor=logprob))


@Factory.register('ChainModelV3')
class ChainModelV3(ModelBase):
    def __init__(self):
        self.node_list = [
            {"name": "C", "causal_func": "clouds"},
            {"name": "G", "causal_func": "gamma"},
            {"name": "B", "causal_func": "blur"},
            {"name": "L", "causal_func": "lensdist"},
            {"name": "DB", "causal_func": "directional_blur"},
            {"name": "D", "causal_func": "defocus"},
            {"name": "N", "causal_func": "noise"}
        ]
        self.alt = [["root", "G"], ["G", "C"], ["C", "DB"], ["DB", "L"], ["L", "N"]],
        self.edge_list = [["root", "C"], ["C", "B"], ["B", "G"], ["G", "L"], ["L", "DB"], ["DB", "D"], ["D", "N"]]

        for node in self.node_list:
            node["corruption_func"] = node["causal_func"]
            node["causal_func"] = getattr(self, node["causal_func"])
            assert node["causal_func"] is not None

        super().__init__()
    
    def clouds(self, scene=None, **kwargs):
        d = DiscreteUniform(n_bins=100)
        v, lp = d.sample()
        p_cloud = 0.75
        if v > p_cloud:
            scale = 0.3
            du = DiscreteUniform(n_bins=100)
            val, logprob = du.sample()
            val *= scale
            logprob += np.log(p_cloud)
        else:
            val = 0.
            logprob = np.log(0.25)
        return dict(factor=val, logprob=dict(factor=logprob))
    
    def blur(self, scene=None, **kwargs):
        factor, = get_values(kwargs, ['factor'], "B", self.edge_list)
        if factor > 0.2:
            sz = 1
            logprob = 0.
        else:
            max_kernel_size = 9
            hn = HalfNormal(scale=max_kernel_size // 2)
            sz, logprob = hn.sample()
            sz = int(sz)
            sz += 1
        return dict(size_x=sz, size_y=sz,
                    logprob=dict(size_x=logprob, size_y=logprob))
    
    def gamma(self, scene=None, **kwargs):
        size_x, size_y = get_values(kwargs, ['size_x', 'size_y'], "G", self.edge_list)
        hn = HalfNormal(scale=0.5)
        val, logprob = hn.sample()
        if size_x <= 3:
            gamma = 0.1 * val
        elif size_x > 3:
            gamma = val
        else:
            gamma = 0.
            logprob = 0
        gamma += 1.
        return dict(gamma=gamma, logprob=dict(gamma=logprob))
    
    def lensdist(self, scene=None, **kwargs):
        gamma, = get_values(kwargs, ['gamma'], "L", self.edge_list)
        hn = HalfNormal(scale=0.5)
        val, logprob = hn.sample()
        if gamma > 1.2:
            distort = 0.05 * val
            p_distort = logprob
            dispersion = 0
            p_dispersion = 0.
        elif gamma > 1.0:
            dispersion = 0.5 * val
            p_dispersion = logprob
            distort = 0.
            p_distort = 0.
        else:
            distort = 0
            p_distort = 0.
            dispersion = 0.
            p_dispersion = 0.
        fit = distort > 0
        return dict(distort=distort, dispersion=dispersion, projector=False, jitter=False, fit=fit,
                    logprob=dict(distort=p_distort, dispersion=p_dispersion))
    
    def directional_blur(self, scene=None, **kwargs):
        distort, = get_values(kwargs, ['distort'], "DB", self.edge_list)
        hn = HalfNormal(scale=0.5)
        val, logprob = hn.sample()
        val2, logprob2 = hn.sample()
        if distort == 0:
            zoom = 0.1 * val
            distance = 0.05 * val2
        else:
            zoom = 0
            logprob = 0.
            distance = 0
        return dict(iterations=12, wrap=False, center_x=0.5, center_y=0.5, distance=distance, angle=0, spin=0,
                    zoom=zoom,
                    logprob=dict(zoom=logprob, distance=logprob2))
    
    def defocus(self, scene=None, **kwargs):
        zoom, distance = get_values(kwargs, ['zoom', 'distance'], "D", self.edge_list)
        if zoom == 0 or distance == 0:
            hn = HalfNormal(scale=3)
            z, z_logprob = hn.sample()
            z += 1.  # z \in {1, 10}
            cat2 = Categorical(n_bins=64)
            f_stop, f_stop_logprob = cat2.sample()
            f_stop += 64  # f_stop \in {64,...,128}
        else:
            z = 1
            f_stop = 128
            z_logprob = 0.
            f_stop_logprob = 0.
        return dict(z=z, f_stop=f_stop,
                    logprob=dict(z=z_logprob, f_stop=f_stop_logprob))
    
    def noise(self, scene=None, **kwargs):
        z, f_stop = get_values(kwargs, ['z', 'f_stop'], "N", self.edge_list)
        hn = HalfNormal(scale=0.5)
        val, logprob = hn.sample()
        if z < 4 or f_stop > 100:
            factor = 0.2 * val
        else:
            factor = 0.05 * val
        return dict(factor=factor, logprob=dict(factor=logprob))
