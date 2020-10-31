# Run inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import ssl
from inference import infer_image


image_np = np.array(Image.open('./data/kite.jpg'))

ssl._create_default_https_context = ssl._create_unverified_context
#model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")
#model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
#model = hub.load('saved_models/ssd_mobilenet_v2_2')
model = hub.load("https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1")
#model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d7/1")
#model = hub.load("https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1")
#model = hub.load("https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1")
#model = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1")

# Check the model's input signature
print("input signature", model.signatures['serving_default'].inputs[0])
# Tensor("input_tensor:0", shape=(1, None, None, 3), dtype=uint8)

# Run inference
image_np = infer_image(image_np, model)

Image.fromarray(image_np).save('detect-test.jpg')