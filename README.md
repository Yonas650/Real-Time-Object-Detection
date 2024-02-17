# Real-Time Object Detection System

This project implements a real-time object detection system using the SSD MobileNet V2 FPNLite 320x320 model, optimized for efficiency and speed, making it suitable for applications requiring real-time performance on limited computational resources. The model is pre-trained on the COCO 2017 dataset, allowing for the detection of a wide variety of objects with high accuracy.

## Features

- **Real-Time Detection**: Leverages the SSD MobileNet V2 FPNLite model for fast object detection.
- **COCO Dataset Labels**: Utilizes the complete COCO label map for recognizing and labeling 80 different object types.
- **Optimized for TPU**: Model performance is optimized for Tensor Processing Units (TPU), but it's also compatible with CPU and GPU setups.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy

### Installation

1. **Clone the repository**

   ```sh
   git clone https://your-repository-url.git
   cd Real-Time-Object-Detection-System
2. **install dependencies**

```pip install tensorflow opencv-python numpy```

3. **Download the SSD MobileNet V2 FPNLite model**
The SSD MobileNet V2 FPNLite model, optimized for the COCO dataset and TPU inference, is too large to include directly in this repository. Follow the instructions below to download and set up the model:

- Visit the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and download the ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz model file.

- Extract the downloaded file into the project directory:

```tar -xvf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz```
This will create a ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 directory containing the model files in your project folder.

4. **Ensure the COCO label map is in place**

For object recognition and labeling, the mscoco_complete_label_map.pbtxt file is required. If it's not already in your project directory, download it using:

```curl -o mscoco_complete_label_map.pbtxt https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_complete_label_map.pbtxt```

5. **Running the Detection System**

To start the object detection system, run the main script:

```python model.py```
This will activate your webcam and begin detecting objects in real-time. Press q to quit the detection window.

6. **Model Information**

- Model Architecture: SSD MobileNet V2 FPNLite
- Input Size: 320x320 pixels
- Training Dataset: COCO 2017
- Optimization: Optimized for TPU (compatible with CPU and GPU)
**Label Map**

The COCO label map included with this project maps numerical class IDs to human-readable labels, covering 80 object categories such as person, bicycle, car, and many more.

**Acknowledgments**
- TensorFlow Object Detection API for providing the model and training data.
- The COCO dataset for the extensive labeled dataset used for training the model.