import torch
from yolov3 import YOLOv3_Detector
from config import YOLO_Param



class Detector:
    """_summary_

    Returns:
        _type_: _description_
    """
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def detect(self, image_array):
        output = self._model.detect(image_array)
        return output

    def __del__(self):
        del self
        torch.cuda.empty_cache()



class Factory:
    """_summary_

    Raises:
        RuntimeError: _description_
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    @staticmethod
    def build_param(model_name):
        if model_name.lower().startswith('yolo'):
            return YOLO_Param
        raise RuntimeError('now only support YOLO')

    @staticmethod
    def build_model(model_name, weight_path, param):
        if model_name.lower() == 'yolov3':
            return YOLOv3_Detector(weight_path, param)
        raise RuntimeError('now only support YOLOv3')
