# Run inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2 
import ssl
from inference import infer_image
import time
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('url', 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2', 'hub url for model or path to model')

def main(argv):
    ssl._create_default_https_context = ssl._create_unverified_context
    model = hub.load(FLAGS.url)

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        image_np = frame
        
        # infer frame
        start_time = time.time()
        result = infer_image(image_np, model)
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)

        # Display the resulting frame
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(main)