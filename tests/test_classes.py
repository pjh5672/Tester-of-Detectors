from unittest import TestCase

from detector import Detector, Factory



class Test_classes(TestCase):
    def test_init(self):
        model_name = 'yolov3'
        param = Factory.build_param(model_name=model_name)
        self.assertRaises(Exception)

    def test_num_classes(self):
        model_name = 'yolov3'
        param = Factory.build_param(model_name=model_name)
        model = Factory.build_model(model_name=model_name, weight_path=None, param=param)
        self.assertEqual(param.num_classes, len(model.class_list))
