# ONNX runtime with object detection

Run detections on custom trained onnx models.
[ONNX Tutorials](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/yoloV3_object_detection_onnxruntime_inference.ipynb) \
EfficientDet implementation based from: https://github.com/murdockhou/Yet-Another-EfficientDet-Pytorch-Convert-ONNX-TVM \
YOLOv3 implementation based from: https://github.com/AlexeyAB/darknet 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install requirements.txt
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
