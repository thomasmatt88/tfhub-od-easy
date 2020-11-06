import tensorflow as tf
import numpy as np

# object_detection.utils.label_map_util
from object_detection.utils.label_map_util import create_category_index, _validate_label_map, load_labelmap, convert_label_map_to_categories, create_categories_from_labelmap, create_category_index_from_labelmap

#object_detection.utils.ops
from object_detection.utils.ops import reframe_box_masks_to_image_masks

# object_detection.utils.visualization_utils
from object_detection.utils.visualization_utils import STANDARD_COLORS, draw_keypoints_on_image, draw_keypoints_on_image_array, draw_bounding_box_on_image, draw_bounding_box_on_image_array, draw_mask_on_image_array, visualize_boxes_and_labels_on_image_array, _get_multiplier_for_color_randomness

COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),
 (0, 2),
 (1, 3),
 (2, 4),
 (0, 5),
 (0, 6),
 (5, 7),
 (7, 9),
 (6, 8),
 (8, 10),
 (5, 6),
 (5, 11),
 (6, 12),
 (11, 12),
 (11, 13),
 (13, 15),
 (12, 14),
 (14, 16)]

def infer_image(image_np, model, show_keypoints):
    """accepts and returns image as numpy array"""

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './data/mscoco_label_map.pbtxt'
    category_index = create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

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
    # output of Mask R-CNN allows instance segmentation
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = reframe_box_masks_to_image_masks(output_dict['detection_masks'], output_dict['detection_boxes'],image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    # Use keypoints if available in detections
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in output_dict and show_keypoints:
        keypoints = output_dict['detection_keypoints']
        keypoint_scores = output_dict['detection_keypoint_scores']

    # Visualization of the results of a detection.
    visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8,
        max_boxes_to_draw=200,
        min_score_thresh=0.30,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

    return image_np