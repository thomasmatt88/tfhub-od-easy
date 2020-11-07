# Run inference on the TF-Hub module.
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tensorflow_hub as hub
import numpy as np
import cv2 
import ssl
from inference import infer_image
import time
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('url', 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2', 'hub url for model or path to model')
flags.DEFINE_string('video', './data/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('show_keypoints', False, 'dont show keypoints by default')

def main(argv):
    ssl._create_default_https_context = ssl._create_unverified_context
    model = hub.load(FLAGS.url)

    video_path = FLAGS.video
    
    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)
    
    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while(True):
        # Capture frame-by-frame
        ret, frame = vid.read()
        image_np = frame
        
        # infer frame
        start_time = time.time()
        result = infer_image(image_np, model, FLAGS.show_keypoints)
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)

        # Display the resulting frame
        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        # Save the resulting frame
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # When everything done, release the capture
    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(main)