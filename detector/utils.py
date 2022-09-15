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


def box_transform_xcycwh_to_x1y1x2y2(bboxes):
    x1y1 = bboxes[:, :2] - bboxes[:, 2:] / 2
    x2y2 = bboxes[:, :2] + bboxes[:, 2:] / 2
    x1y1x2y2 = np.concatenate((x1y1, x2y2), axis=1)
    x1y1x2y2 = x1y1x2y2.clip(min=0., max=1.)
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


def run_NMS_for_YOLO(prediction, iou_threshold=0.5, maxDets=100):
    bboxes = prediction[:, 1:5] * 100
    scores = prediction[:, 5]

    if len(bboxes) == 0:
        return []

    if bboxes.dtype.kind == "i":
        bboxes = bboxes.astype("float")

    x1 = np.maximum(bboxes[:, 0] - bboxes[:, 2]/2, 0)
    y1 = np.maximum(bboxes[:, 1] - bboxes[:, 3]/2, 0)
    x2 = bboxes[:, 0] + bboxes[:, 2]/2
    y2 = bboxes[:, 1] + bboxes[:, 3]/2
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    pick = []
    while len(order) > 0:
        i = order[0]
        pick.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h)
        ious = overlap / (areas[i] + areas[order[1:]] - overlap + 1e-8)
        order = order[np.where(ious <= iou_threshold)[0] + 1]
    return prediction[pick[:maxDets]]
