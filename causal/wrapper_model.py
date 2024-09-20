# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import logging
from pathlib import Path

from .registry import Factory


@Factory.register('WrapperModel')
class WrapperModel(object):
    '''
        Wraps multiple DAGs into a single class
    '''
    def __init__(self, dag_classes, config_files=None, dags=None, **kwargs):
        '''
            Initializes multiple DAGs.

            Parameters:
                dag_classes: A list of DAG Class names to create
                config_files: A list of config files associated with each class
                    to create a DAG from
                dags: A list of already initialized DAGs. If given will instead
                    wrap the given DAGs instead of creating new ones
        '''
        # If DAGs already provided 
        if dags is not None:
            if isinstance(dags, list):
                self.dags = dags
            else:
                self.dags = [dags]
        else:
            if not isinstance(dag_classes, list):
                dag_classes = [dag_classes]
            if not isinstance(config_files, list):
                config_files = [config_files]
            assert len(dag_classes) == len(config_files)

            self.dags = []
            for dag_cls, config_file in zip(dag_classes, config_files):
                if config_file is not None:
                    _kwargs = dict(config_file=config_file, **kwargs)
                else:
                    _kwargs = kwargs
                logging.info(f'Creating {dag_cls} instance with args: {_kwargs}')
                self.dags.append(Factory.create_class(dag_cls, **kwargs))

        self.n_dags = len(self.dags)

    @classmethod
    def load(cls, dag_classes, config_files):
        dags = []
        for dag_cls, config_file in zip(dag_classes, config_files):
            logging.info(f'Loading {dag_cls} from config file {config_file}')
            dags.append(Factory.create_func(dag_cls).load(config_file))
        return cls(None, None, dags=dags)

    def save(self, directory):
        assert not Path(directory).is_file()
        Path(directory).mkdir(parents=True, exist_ok=True)
        ext = 'yaml'

        for i, dag in enumerate(self.dags):
            dag.save(f'{directory}/dag_{i}.{ext}')

    def sample(self, scene=None, idx=0): 
        sampled = self.dags[idx].sample(scene=scene)

        return sampled

    def update_tree(self, *args, idx=0, **kwargs):
        self.dags[idx].update_tree(*args, **kwargs) 
