# <div align="center">Object Detector Tester</div>

## [Description]

This is repository of source code for testing trained object detection models. Currently only supports trained YOLOv3 model architecture.


## [Usage]

### 1. Installation
```bash
$ pip install -r requirements.txt
```

### 2. Build **`parameter`**
 - **parameter** 
    - input_size: input size calculated by detection model
    - num_classes: number of prediction classes in detection model
    - conf_threshold: minimum confidence threshold to filter detection results
    - nms_threshold: minimum IoU threshold for Non-Maximum Suppresion
    - max_dets: maximum number of final detected results in a single image
    - device: computational device setting (-1: for CPU, >=0: for GPU(gpu id))
    - anchors: setting the anchor box size
    - class_list: category name corresponding to the detection class of the model
 - **model**
    - detector model architecture for deploying

```python
from detector import Factory

model_name = 'yolov3'
weight_path = '../weights/model_EP035.pt'
parameter = Factory.build_param(model_name=model_name)
parameter.device = 0  
model = Factory.build_model(model_name=model_name, weight_path=weight_path, param=parameter)
```
 
### 3. Set up **`detector`**

```python
from detector import Detector

detector = Detector()
detector.model = model
```

### 4. Run
 - image frame should be on RGB color space
 - pred_yolo 
    - prediction output for labeling data format for automatic collection of training data
    - [num_objects, YOLO format] sized matrix, YOLO format of **(class_id, norm_xc, norm_yc, norm_w, norm_h, confidence score)**
 - pred_voc
    - prediction output for displaying bounding box on the original image
    - [num_objects, VOC format] sized matrix, VOC format of **(class_id, image_x1, image_y1, image_x2, image_y2, confidence score)**

```python
image_path = './samples/image.jpg'
frame = cv2.imread(image_path)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pred_yolo, pred_voc = detector.detect(frame) 
```

 - **Detection Result** 
   - original iamge(left) and detection result(right) from early stopped model
    <div align="center">
    <a href=""><img src=./asset/image.jpg width="30%" /></a>
    <a href=""><img src=./asset/result.jpg width="30%" /></a>
    </div>


## [Update]

<details>
    <summary><b> Timeline in 2022 </b></summary>

| Date | Content |
|:----:|:-----|
| 08-24 | first commit |

</details>


## [Contact]
- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  
