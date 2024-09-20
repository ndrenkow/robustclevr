# Copyright 2024, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class Factory:
    """ The factory class for creating executors"""

    registry = {}
    """ Internal registry for available executors """

    @classmethod
    def register(cls, executor_name: str) -> Callable:
        """ Class method to register Executor class to the internal registry.
        Args:
            executor_name (str): The name of the executor.
        Returns:
            The Executor class itself.
        """

        def inner_wrapper(wrapped_class) -> Callable:
            if executor_name in cls.registry:
                logger.warning('Executor %s already exists. Will replace it', executor_name)
            cls.registry[executor_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    # end register()

    @classmethod
    def create_class(cls, class_name: str, **kwargs):
        """ Factory command to create the executor.
        This method gets the appropriate Executor class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.
        Args:
            class_name (str): The name of the executor to create.
        Returns:
            An instance of the executor that is created.
        """

        if class_name not in cls.registry:
            logger.warning('%s does not exist in the registry', class_name)
            return None

        exec_class = cls.registry[class_name]
        executor = exec_class(**kwargs)
        return executor

    @classmethod
    def create_func(cls, func_name: str):
        """ Factory command to create the executor.
        This method gets the appropriate function from the registry and
        returns it, so that it can be called by whatever invoked this funciton.
        Args:
            func_name (str): The name of the function.
        Returns:
            A callable object corresponding to the desired function.
        """

        if func_name not in cls.registry:
            logger.warning('%s does not exist in the registry', func_name)
            return None

        return cls.registry[func_name]
