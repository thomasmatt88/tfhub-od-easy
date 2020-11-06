# tfhub-od-easy
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

[TensorFlow Hub](https://tfhub.dev/) lets you search and discover hundreds of trained, ready-to-deploy machine learning models in one place. 
This repo offers a minimal codebase to deploy **TF2 Image Object Detection** models from TensorFlow Hub for image object detection, 
video object detection, webcam object detection, and evaluation on the MS COCO 2017 validation dataset (5000 images). Simply replace the *--url* argument 
in any of the demos below with the url of the model of interest from https://tfhub.dev/s?module-type=image-object-detection&tf-version=tf2.  
  
Example urls:
* https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2
* https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1
* https://tfhub.dev/tensorflow/efficientdet/d3/1
* etc.


```
pip install -r requirements.txt
```

### Detect Image Demo
``` bash
# detect image with SSD Mobilenet v2 
python detect_image.py --url https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2 --image_input ./data/kite.jpg

# detect image with EfficientDet-d7
python detect_image.py --url https://tfhub.dev/tensorflow/efficientdet/d7/1 --image_input ./data/kite.jpg
```

### Detect Video Demo
``` bash
# detect video with SSD Mobilenet v2 
python detect_video.py --url https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2 --video ./data/video.mp4 --output ./detect-test.mp4
```

### Detect Webcam Demo
``` bash
# detect webcam with SSD Mobilenet v2 
python detect_video.py --url https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2 --video 0
```

### Acknowledgements
This project was inspired by
* [Yolov4 tf2](https://github.com/hunglc007/tensorflow-yolov4-tflite)
* [Yolov4 tf2 with webcam detection](https://github.com/theAIGuysCode/tensorflow-yolov4-tflite)
* [Object Detection API Demo](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb)
