import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response

app = Flask(__name__)

#load the object detection model
model = tf.saved_model.load('./ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')

#load class names
class_names = {}
current_id = None
with open('./mscoco_complete_label_map.pbtxt', 'r') as f:
    for line in f:
        if "id:" in line:
            current_id = int(line.strip().split(' ')[-1])
        if "display_name:" in line:
            display_name = line.strip().split('"')[1]
            class_names[current_id] = display_name

#function to draw boxes
def draw_boxes(frame, boxes, classes, scores, min_score_thresh=.5):
    for i in range(boxes.shape[1]):
        if scores[0, i] > min_score_thresh:
            box = boxes[0, i] * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
            class_id = classes[0, i]
            class_name = class_names.get(class_id, 'N/A')
            cv2.rectangle(frame, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (int(box[1]), int(box[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#video streaming generator function
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #object detection
        input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)
        detections = model(input_tensor)

        #extracting detection results
        boxes = detections['detection_boxes'].numpy()
        classes = detections['detection_classes'].numpy().astype(np.int32)
        scores = detections['detection_scores'].numpy()

        #draw boxes on the frame
        draw_boxes(frame, boxes, classes, scores)

        #encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        #yield the frame in the format that Flask expects
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
