# <div align="center">Object Detector Tester</div>

---

## [Contents]  
1. [Description](#description)  
2. [Usage](#usage)  
  2-1. [Install](#install)  
  2-2. [Build **`Parameter`**](#build-parameter)  
  2-3. [Set up **`Detector`**](#set-up-detector)  
  2-4. [Detect](#detect)  
3. [Update](#update)  
4. [Contact](#contact)  

---

## [Description]

This is repository of source code for testing trained object detection models. Currently only supports trained YOLOv3 model architecture.


## [Usage]

### Install
```bash
$ pip install -r requirements.txt
```

### Build **`Parameter`**
 - **Parameter** 
    - **`input_size`**: input size calculated by detection model
    - **`num_classes`**: number of prediction classes in detection model
    - **`conf_threshold`**: minimum confidence threshold to filter detection results
    - **`nms_threshold`**: minimum IoU threshold for Non-Maximum Suppresion
    - **`max_dets`**: maximum number of final detected results in a single image
    - **`device`**: computational device setting (-1: for CPU, >=0: for GPU(gpu id))
    - **`anchors`**: setting the anchor box size
 - **Model**
    - detector model architecture for deploying

```python
from detector import Factory

model_name = 'yolov3'
weight_path = './weights/voc_best.pt'
parameter = Factory.build_param(model_name=model_name)
parameter.device = 0  
model = Factory.build_model(model_name=model_name, weight_path=weight_path, param=parameter)
```
 
### Set up **`Detector`**

```python
from detector import Detector

detector = Detector()
detector.model = model
```


### Detect
 - Image frame should be on RGB color space
 - Return(index 0) - **pred_yolo** 
    - prediction output for labeling data format for automatic collection of training data
    - [num_objects, YOLO format] sized matrix, YOLO format of `(class_id, norm_xc, norm_yc, norm_w, norm_h, confidence_score)`
 - Return(index 1) - **pred_voc**
    - prediction output for displaying bounding box on the original image
    - [num_objects, VOC format] sized matrix, VOC format of `(class_id, image_x1, image_y1, image_x2, image_y2, confidence_score)`
#####

 - **Sample Command**
   ```python
   image_path = './samples/image.jpg'
   frame = cv2.imread(image_path)
   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   pred_yolo, pred_voc = detector.detect(frame)
   ```

 - **Detection Result** 
   <div align="center">
      <a href=""><img src=./asset/image.jpg width="40%" /></a>
      <a href=""><img src=./asset/result.jpg width="40%" /></a>
   </div>
   <div align="center">
      - Original iamge(left) and detection result(right) from model trained on Pascal VOC Dataset
   </div>



## [Update]

<details>
    <summary><b> Timeline in 2022 </b></summary>

| Date | Content |
|:----:|:-----|
| 09-15 | first code cleansing |
| 08-24 | first commit |

</details>


## [Contact]
- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  
