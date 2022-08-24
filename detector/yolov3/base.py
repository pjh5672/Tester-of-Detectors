import abc
import numpy as np


class Detector(metaclass=abc.ABCMeta):
    """_summary_

    Args:
        metaclass (_type_, optional): _description_. Defaults to abc.ABCMeta.
    """
    @abc.abstractmethod
    def detect(self, image: np.ndarray):
        """[summary]
        Args:
            image: [description]
        Raises:
            NotImplementedError: [description]
        """
        pass
