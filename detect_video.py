# Run inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# object_detection.utils.label_map_util
from object_detection.utils.label_map_util import create_category_index, _validate_label_map, load_labelmap, convert_label_map_to_categories, create_categories_from_labelmap, create_category_index_from_labelmap

#object_detection.utils.ops
from object_detection.utils.ops import reframe_box_masks_to_image_masks

# object_detection.utils.visualization_utils
from object_detection.utils.visualization_utils import STANDARD_COLORS, draw_keypoints_on_image, draw_keypoints_on_image_array, draw_bounding_box_on_image, draw_bounding_box_on_image_array, draw_mask_on_image_array, visualize_boxes_and_labels_on_image_array, _get_multiplier_for_color_randomness

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './data/mscoco_label_map.pbtxt'
category_index = create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

import cv2 
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    image_np = frame
    image = np.asarray(image_np)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = reframe_box_masks_to_image_masks(output_dict['detection_masks'], output_dict['detection_boxes'],image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    # Visualization of the results of a detection.
    visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    # Display the resulting frame
    result = image_np
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
