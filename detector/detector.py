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
        if model_name.lower() == 'yolov3':
            return YOLO_Param
        raise RuntimeError('now only support YOLOv3')

    @staticmethod
    def build_model(model_name, weight_path, param):
        if model_name.lower() == 'yolov3':
            return YOLOv3_Detector(weight_path, param)
        raise RuntimeError('now only support YOLOv3')



if __name__ == '__main__':
    import os
    from pathlib import Path

    import cv2
    from visualize import generate_random_color, visualize

    image_dir = 'd:/MS-COCO/coco2017/images/train/'
    image_dir = Path(image_dir)
    image_paths = [str(image_dir / fn) for fn in os.listdir(image_dir) if fn.lower().endswith(('png', 'jpg', 'jpeg'))]

    model_name = 'yolov3'
    # weight_path = './weights/model_EP035.pt'
    weight_path = None

    param = Factory.build_param(model_name=model_name)
    param.device = -1
    model = Factory.build_model(model_name=model_name, weight_path=weight_path, param=param)

    detector = Detector()
    detector.model = model

    class_list = model.class_list
    color_list = generate_random_color(param.num_classes)
    index = 0
    filename = image_paths[index].split(os.sep)[-1]
    image = cv2.imread(image_paths[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred_yolo, pred_voc = detector.detect(image)
    canvas = visualize(image, pred_voc, class_list, color_list, True, True) if len(pred_voc) > 0 else image

    cv2.imshow(filename, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    del detector
