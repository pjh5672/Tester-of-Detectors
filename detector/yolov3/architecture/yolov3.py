from torch import nn

from yolov3_modules import Darknet53_backbone, YOLOv3_FPN, YOLOv3_head



class YOLOv3_Model(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_size, num_classes, anchors):
        super().__init__()
        self.backbone = Darknet53_backbone()
        self.fpn = YOLOv3_FPN()
        self.head = YOLOv3_head(input_size=input_size, num_classes=num_classes, anchors=anchors)


    def forward(self, x):
        x1, x2, x3 = self.backbone(x)
        out_l, out_m, out_s = self.fpn(x1, x2, x3)
        predictions = self.head(out_l, out_m, out_s)
        return predictions
