from dataclasses import dataclass


@dataclass
class YOLO_Param:
    """_summary_
    """
    input_size = 416
    num_classes = 80
    conf_threshold = 0.5
    nms_threshold = 0.5
    max_dets = 100
    device = -1

    anchors = [
        [[10, 13], [16, 30], [33, 23]], # anchor_S
        [[30, 61], [62, 45], [59, 119]], # anchor_M
        [[116, 90], [156, 198], [373, 326]], # anchor_L
    ]
