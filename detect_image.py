# Run inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import ssl
from inference import infer_image
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('url', 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2', 'hub url for model or path to model')
flags.DEFINE_string('image_input', './data/kite.jpg', 'path to image')
flags.DEFINE_string('image_output', './detect-test.jpg', 'path to image')
flags.DEFINE_boolean('show_keypoints', False, 'dont show keypoints by default')

def main(argv):
    image_np = np.array(Image.open(FLAGS.image_input))

    # handle grayscale image
    if image_np.ndim == 2:
        image_np = np.repeat(image_np[..., np.newaxis], 3, axis=2)

    ssl._create_default_https_context = ssl._create_unverified_context
    model = hub.load(FLAGS.url)

    # Check the model's input signature
    print("input signature", model.signatures['serving_default'].inputs[0])
    # Tensor("input_tensor:0", shape=(1, None, None, 3), dtype=uint8)

    # Run inference
    image_np = infer_image(image_np, model, FLAGS.show_keypoints)

    Image.fromarray(image_np).save(FLAGS.image_output)

if __name__ == '__main__':
    app.run(main)
