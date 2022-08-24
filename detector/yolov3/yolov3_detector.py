import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from base import Detector
from architecture import YOLOv3_Model
from utils import *



class YOLOv3_Detector(Detector):
	"""_summary_

	Args:
		Detector (_type_): _description_
	"""
	def __init__(self, weight_path, param):
		super().__init__()
		self.input_size = param.input_size
		self.model = YOLOv3_Model(input_size=self.input_size, num_classes=param.num_classes, anchors=param.anchors)
		assert param.device <= torch.cuda.device_count()-1, \
			f'can not assign cuda:{param.device}, available cuda device numbers:{torch.cuda.device_count()}'
		self.device =  torch.device(f'cuda:{param.device}' if torch.cuda.is_available() and param.device > -1 else 'cpu')
		if (weight_path is not None) and (Path(weight_path).exists()):
			self.model, self.class_list = load_model(self.model, weight_path, self.device)
		else:
			self.model, self.class_list = self.model.eval().to(self.device), param.class_list
		self.conf_thresh = param.conf_threshold
		self.nms_thresh = param.nms_threshold
		self.max_dets = param.max_dets


	@torch.no_grad()
	def detect(self, image):
		img_h, img_w, _ = image.shape
		tensor_image, max_side = preprocess(image, self.input_size)
		output = self.model(tensor_image.to(self.device))
		output = torch.cat(output, dim=1)[0]
		output[..., 4:] = torch.sigmoid(output[..., 4:])
		output[..., 5:] *= output[..., 4:5]
		pred_yolo = output.cpu().numpy()
		pred_yolo[:, :4] = clip_box_coordinates(bboxes=pred_yolo[:, :4]/self.input_size)
		pred_yolo = filter_obj_score(prediction=pred_yolo, conf_threshold=self.conf_thresh)
		pred_yolo = run_NMS_for_YOLO(prediction=pred_yolo, iou_threshold=self.nms_thresh, maxDets=self.max_dets)

		if len(pred_yolo) > 0:
			pred_voc = pred_yolo.copy()
			pred_voc[:, 1:5][:,[0,2]] *= (max_side/img_w)
			pred_voc[:, 1:5][:,[1,3]] *= (max_side/img_h)
			pred_voc[:, 1:5] = box_transform_xcycwh_to_x1y1x2y2(pred_voc[:, 1:5])
			pred_voc[:, 1:5] = scale_to_original(pred_voc[:, 1:5], scale_w=img_w, scale_h=img_h)
			return (pred_yolo, pred_voc)
		return ([], [])


	def __del__(self):
		del self
		torch.cuda.empty_cache()
