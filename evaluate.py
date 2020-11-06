from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import ssl

# object_detection.utils.label_map_util
from object_detection.utils.label_map_util import create_category_index_from_labelmap

flags.DEFINE_string('annotation_path', "./data/dataset/val2017.txt", 'annotation path')
flags.DEFINE_string('write_image_path', "./data/detection/", 'write image path')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def main(_argv):
    # small discrepencies in index of mscoco_label_map.pbtxt and coco.names
    # order of labels are the same but not the indices
    PATH_TO_LABELS = './data/mscoco_label_map.pbtxt'
    CATEGORY_INDEX = create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    CLASSES = read_class_names('./data/classes/coco.names')

    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    #detected_image_path = "./data/detection/"
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    #if os.path.exists(detected_image_path): shutil.rmtree(detected_image_path)

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    #os.mkdir(detected_image_path)

    # Build Model
    ssl._create_default_https_context = ssl._create_unverified_context
    model = hub.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')
    #model = hub.load('https://tfhub.dev/tensorflow/efficientdet/d7/1')
    infer = model.signatures['serving_default']

    num_lines = sum(1 for line in open("./data/dataset/val2017.txt"))
    with open("./data/dataset/val2017.txt", 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            # Ground Truth
            annotation = line.strip().split()
            image_path = annotation[0]
            print("image_path", image_path)
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)

            
            # Predict 
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            image_np = np.array(Image.open(image_path))
            image = np.asarray(image_np)
            # handle grayscale images
            if image.ndim == 2:
                image = np.repeat(image[..., np.newaxis], 3, axis=2)
            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis,...]
            output_dict = infer(input_tensor)

            
            valid_detections = output_dict['num_detections']
            classes = output_dict['detection_classes']
            boxes = output_dict['detection_boxes']
            scores = output_dict['detection_scores']

            selected_indices = tf.image.non_max_suppression(boxes[0], scores[0], max_output_size=50, iou_threshold=FLAGS.iou, score_threshold=FLAGS.score)
            selected_boxes = tf.gather(boxes[0], selected_indices)
            selected_scores = tf.gather(scores[0], selected_indices)
            selected_classes = tf.gather(classes[0], selected_indices)
            

            valid_detections = selected_scores.numpy().shape[0]

            with open(predict_result_path, 'w') as f:
                image_h, image_w, _ = image.shape # numpy dimensions
                for i in range(int(valid_detections)):
                    if int(selected_classes[i]) < 0 or int(selected_classes[i]) > 80: continue # only include 80 and below
                
                    # normalized coordinates
                    ymin, xmin, ymax, xmax = selected_boxes[i].numpy()
                    # rescale
                    ymin = ymin*image_h
                    ymax = ymax*image_h
                    xmin = xmin*image_w
                    xmax = xmax*image_w

                    score = selected_scores[i]
                    class_ind = int(selected_classes[i])
                    class_name = CATEGORY_INDEX[class_ind]['name']
                    score = '%.4f' % score
                    bbox_mess = ' '.join([class_name, score, str(xmin), str(ymin), str(xmax), str(ymax)]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print(num, num_lines)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


