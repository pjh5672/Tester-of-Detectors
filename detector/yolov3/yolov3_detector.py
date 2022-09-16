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
	def __init__(self, weight_path, parameter):
		super().__init__()
		self.input_size = parameter.input_size
		self.model = YOLOv3_Model(input_size=self.input_size, num_classes=parameter.num_classes, anchors=parameter.anchors)
		assert parameter.device <= torch.cuda.device_count()-1, \
			f'can not assign cuda:{parameter.device}, available cuda device numbers:{torch.cuda.device_count()}'
		self.device =  torch.device(f'cuda:{parameter.device}' if torch.cuda.is_available() and parameter.device > -1 else 'cpu')

		if (weight_path is not None) and (Path(weight_path).exists()):
			self.model, self.class_list = load_model(self.model, weight_path, self.device)
		else:
			self.model = self.model.eval().to(self.device)
			self.class_list = {}
			for k, v in enumerate(range(parameter.num_classes)):
				self.class_list[k] = v

		self.conf_thresh = parameter.conf_threshold
		self.nms_thresh = parameter.nms_threshold
		self.max_dets = parameter.max_dets


	@torch.no_grad()
	def detect(self, image):
		img_h, img_w, _ = image.shape
		tensor, max_size = preprocess(image, self.input_size)
		predictions = self.model(tensor.unsqueeze(dim=0).to(self.device))
		predictions = torch.cat(predictions, dim=1)
		predictions[..., 4:] = torch.sigmoid(predictions[..., 4:])
		predictions[..., 5:] *= predictions[..., 4:5]

		prediction = predictions[0].cpu().numpy()
		prediction[:, :4] = box_transform_xcycwh_to_x1y1x2y2(prediction[:, :4], clip_max=self.input_size)
		prediction = filter_obj_score(prediction=prediction, conf_threshold=self.conf_thresh)
		prediction = run_NMS(prediction, iou_threshold=self.nms_thresh, maxDets=self.max_dets, class_agnostic=False)

		if len(prediction) > 0:
			prediction[:, 1:5] /= self.input_size
			prediction[:, 1:5][:,[0,2]] *= (max_size/img_w)
			prediction[:, 1:5][:,[1,3]] *= (max_size/img_h)
			prediction[:, 1:5] = scale_to_original(prediction[:, 1:5], scale_w=img_w, scale_h=img_h)
			return prediction
		return []


	def __del__(self):
		del self
		torch.cuda.empty_cache()
