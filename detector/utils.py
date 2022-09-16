import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF


IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation



def load_model(model, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model.to(device), checkpoint['class_list']


def preprocess(image, input_size):
    image, max_size = transform_square_image(image)
    image = cv2.resize(image, dsize=(input_size, input_size))
    tensor = normalize(to_tensor(image))
    return tensor, max_size


def to_tensor(image):
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    image = torch.from_numpy(image).float()
    image /= 255.
    return image


def normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    tensor = TF.normalize(image, mean, std)
    return tensor


def transform_square_image(image):
    pad_h, pad_w = 0, 0
    img_h, img_w, img_c = image.shape
    max_size = max(img_h, img_w)

    if img_h < max_size:
        pad_h = max_size - img_h
    if img_w < max_size:
        pad_w = max_size - img_w

    pad_image = np.zeros(shape=(img_h+pad_h, img_w+pad_w, img_c), dtype=image.dtype)
    pad_image[:img_h, :img_w, :] = image
    return pad_image, max_size


def scale_to_original(bboxes, scale_w, scale_h):
    bboxes[:,[0,2]] *= scale_w
    bboxes[:,[1,3]] *= scale_h
    return bboxes.round(2)


def clip_box_coordinates(bboxes):
    bboxes = box_transform_xcycwh_to_x1y1x2y2(bboxes)
    bboxes = box_transform_x1y1x2y2_to_xcycwh(bboxes)
    return bboxes


def box_transform_xcycwh_to_x1y1x2y2(bboxes, clip_max=None):
    x1y1 = bboxes[:, :2] - bboxes[:, 2:] / 2
    x2y2 = bboxes[:, :2] + bboxes[:, 2:] / 2
    x1y1x2y2 = np.concatenate((x1y1, x2y2), axis=1)
    x1y1x2y2 = x1y1x2y2.clip(min=0., max=clip_max if clip_max is not None else 1.)
    return x1y1x2y2


def box_transform_x1y1x2y2_to_xcycwh(bboxes):
    wh = bboxes[:, 2:] - bboxes[:, :2]
    xcyc = bboxes[:, :2] + wh / 2
    xcycwh = np.concatenate((xcyc, wh), axis=1)
    return xcycwh


def filter_obj_score(prediction, conf_threshold=0.01):
    valid_index = (prediction[:, 4] >= conf_threshold)
    bboxes = prediction[:, :4][valid_index]
    conf_scores = prediction[:, 4][valid_index]
    class_ids = np.argmax(prediction[:, 5:][valid_index], axis=1)
    return np.concatenate([class_ids[:, np.newaxis], bboxes, conf_scores[:, np.newaxis]], axis=-1)


def hard_NMS(bboxes, scores, iou_threshold):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    pick = []

    while len(order) > 0:
        pick.append(order[0])
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[order[0]], x1[order[1:]])
        yy1 = np.maximum(y1[order[0]], y1[order[1:]])
        xx2 = np.minimum(x2[order[0]], x2[order[1:]])
        yy2 = np.minimum(y2[order[0]], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h)
        ious = overlap / (areas[order[0]] + areas[order[1:]] - overlap + 1e-8)
        order = order[np.where(ious <= iou_threshold)[0] + 1]
    return pick


def run_NMS(prediction, iou_threshold, maxDets=100, class_agnostic=False):
    if len(prediction) == 0:
        return []

    if class_agnostic:
        pick = hard_NMS(prediction[:, 1:5], prediction[:, 5], iou_threshold)
        return prediction[pick[:maxDets]]

    prediction_multi_class = []
    for cls_id in np.unique(prediction[:, 0]):
        pred_per_cls_id = prediction[prediction[:, 0] == cls_id]
        pick_per_cls_id = hard_NMS(pred_per_cls_id[:, 1:5], pred_per_cls_id[:, 5], iou_threshold)
        prediction_multi_class.append(pred_per_cls_id[pick_per_cls_id])
    prediction_multi_class = np.concatenate(prediction_multi_class, axis=0)
    order = prediction_multi_class[:, -1].argsort()[::-1]
    return prediction_multi_class[order[:maxDets]]
