# Run inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2 
import ssl
from inference import infer_image

ssl._create_default_https_context = ssl._create_unverified_context
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    image_np = frame
    
    # Display the resulting frame
    result = infer_image(image_np, model)
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
