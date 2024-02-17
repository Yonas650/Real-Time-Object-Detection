import cv2
import numpy as np
import tensorflow as tf


model = tf.saved_model.load('./ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')

#this draws detection boxes
def draw_boxes(frame, boxes, classes, scores, min_score_thresh=.5):
    for i in range(boxes.shape[1]):
        if scores[0, i] > min_score_thresh:
            box = boxes[0, i] * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
            class_id = classes[0, i]
            class_name = class_names.get(class_id, 'N/A')  #telling it to use 'N/A' if class_id is not in class_names
            cv2.rectangle(frame, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (int(box[1]), int(box[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#loading class names to a dictionary
class_names = {}
current_id = None
with open('./mscoco_complete_label_map.pbtxt', 'r') as f:
    for line in f:
        if "id:" in line:
            current_id = int(line.strip().split(' ')[-1])
        if "display_name:" in line:
            display_name = line.strip().split('"')[1]
            class_names[current_id] = display_name

#intializes video capture
cap = cv2.VideoCapture(0)

while True:
    #reads frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    #converting frame to a tensor and pass it through the model
    input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)
    detections = model(input_tensor)

    #getting detection results
    boxes = detections['detection_boxes'].numpy()
    classes = detections['detection_classes'].numpy().astype(np.int32)
    scores = detections['detection_scores'].numpy()

    #drawing detection boxes on the frame
    draw_boxes(frame, boxes, classes, scores)

    #displaying  the frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
