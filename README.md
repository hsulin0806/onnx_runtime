# ONNX runtime with object detection

Run detections on custom trained onnx models.
[ONNX Tutorials](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/yoloV3_object_detection_onnxruntime_inference.ipynb)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install onnx==1.7.0
pip install onnxruntime-gpu==1.3.0
pip install numpy==1.18.4
pip install pillow==7.1.2
pip install matplotlib==3.2.1
```

## Setup

.onnx model
```
/onnx_runtime/models/myOnnxModel.onnx
```
test images
```
/onnx_runtime/data/images
```

## Detection

```
python detect.py <original framework> <path to onnx model> <path to test images>
```
Output:
```
Inference time:  0.18  sec
3 objects identified in source image.
```
![alt text](data/predictions/predictions.jpg)
## Credit
Credit to the tutorials and documents provided by onnx. \
https://github.com/onnx