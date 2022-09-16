import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from detector import Detector, Factory


def load_source(source):
    if os.path.isdir(source):
        image_dir = Path(source)
        frames = [image_dir / fn for fn in os.listdir(image_dir) if fn.lower().endswith(('png', 'jpg', 'jpeg'))]
        frame_info = {
            'run_mode': 0
        }
    elif os.path.isfile(source):
        if source.lower().endswith(('png', 'jpg', 'jpeg')):
            frames = [Path(source)]
            frame_info = {
                'run_mode': 0
            }
        elif source.lower().endswith(('avi', 'mp4')):
            frames = cv2.VideoCapture(source)
            frame_info = {
                'vid_h': int(frames.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'vid_w': int(frames.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'frame_cnt': int(frames.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': round(frames.get(cv2.CAP_PROP_FPS), 0),
                'run_mode': 1
            }
        else:
            raise RuntimeError('this file is neither Image nor Video')
    else:
        raise RuntimeError('this source is neither Directory nor File')
    return frames, frame_info


def imwrite(filename, image):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, image)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def detect_image(filename, file_path, class_list, color_list):
    frame = cv2.imread(file_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = detector.detect(frame)
    canvas = visualize(frame, out, class_list, color_list, True, True)
    imwrite(str(ROOT / f'./example/results/{filename}'), canvas)


def detect_images(dirname, file_paths, class_list, color_list):
    for image_path in file_paths:
        filename = image_path.name
        frame = cv2.imread(str(image_path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = detector.detect(frame)
        canvas = visualize(frame, out, class_list, color_list, True, True)
        imwrite(str(ROOT/f'./example/results/{dirname}/{filename}'), canvas)


def detect_video(vid_name, video, frame_info, class_list, color_list, rec=False, codec='mp4v'):
    fourcc = cv2.VideoWriter_fourcc(*codec) # mp4:'mp4v', avi:'XVID'
    fps, vid_w, vid_h = frame_info['fps'], frame_info['vid_w'], frame_info['vid_h']
    if rec:
        recoder = cv2.VideoWriter(str(ROOT/f'./example/results/videos/{vid_name}'), fourcc, fps, (vid_w, vid_h))

    while (video.isOpened()):
        ret, frame = video.read()
        if ret:
            since = time.time()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = detector.detect(frame)
            text= f'{(time.time() - since)*1000:.0f}ms/image'
            canvas = visualize(frame, out, class_list, color_list, True, True)
            canvas = np.ascontiguousarray(canvas, dtype=np.uint8)
            cv2.putText(canvas, text, (15, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (230, 230, 230), 2)
            cv2.imshow(vid_name, canvas)
            if rec:
                recoder.write(canvas)
            key = cv2.waitKey(1)
            if key == 27:
                break
            if key == ord('s'):
                cv2.waitKey()
    frames.release()
    if rec:
        recoder.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    from detector import visualize, generate_random_color

    source = './example/samples/image.jpg'
    # source = './example/samples/images'
    # source = './example/samples/videos/1280x720_many_motor_1.mp4'
    frames, frame_info = load_source(source)

    model_name = 'yolov3'
    weight_path = './weights/voc_best.pt'

    param = Factory.build_param(model_name=model_name)
    param.device = -1
    param.num_classes = 20
    
    model = Factory.build_model(model_name=model_name, weight_path=weight_path, param=param)

    detector = Detector()
    detector.model = model
    
    class_list = model.class_list
    color_list = generate_random_color(param.num_classes)

    if frame_info['run_mode'] == 0:
        if len(frames) == 1:
            os.makedirs(str(ROOT/'example/results/'), exist_ok=True)
            detect_image(filename=frames[0].name, file_path=str(frames[0]), class_list=class_list, color_list=color_list)
        else:
            dirname = frames[0].parent.name
            os.makedirs(str(ROOT/f'./example/results/{dirname}'), exist_ok=True)
            detect_images(dirname=dirname, file_paths=frames, class_list=class_list, color_list=color_list)
    elif frame_info['run_mode'] == 1:
        vid_name = Path(source).name
        os.makedirs(str(ROOT/f'./example/results/videos'), exist_ok=True)
        detect_video(vid_name, frames, frame_info, class_list, color_list, rec=False, codec='mp4v')
    del detector