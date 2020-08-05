import onnxruntime
import torch
from PIL import Image
import time
import sys
import os
import numpy as np

sys.path.append('utils/')
from yolov3 import get_yolov3_input, yolov3_postprocess, yolov3_display_objdetect_image
from efficientdet import invert_affine, get_efficientdet_input, efficientdet_postprocess, efficientdet_display, \
    ef_preprocess, BBoxTransform, ClipBoxes

# custom classes
classes = ['Vehicle registration plate']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = True
use_float16 = False
threshold = 0.2
iou_threshold = 0.2

# coco classes
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
           'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
           'toothbrush']


def inference(session, framework, images):
    image_pths = os.listdir(images)
    num = 1
    for i in image_pths:
        # get input data
        pth = images + "/" + i
        img = Image.open(pth)

        # YOLOv3
        if framework == "yolov3":
            # get model input
            model_image_size = (416, 416)
            img_data, input_name, img_size = get_yolov3_input(img, session, model_image_size)

            # predict with ONNX runtime
            start = time.time()
            for j in range(10):
                boxes, scores, indices = session.run(None, {input_name: img_data, "image_shape": img_size})

                # postprocess predictions
                out_boxes, out_scores, out_classes, objects_identified = yolov3_postprocess(boxes, scores, indices,
                                                                                            classes)
            end = time.time()
            inference_time = round((end - start) / 10, 2)
            print("Inference time: ", inference_time, " sec")

            # display and save predictions
            yolov3_display_objdetect_image(img, out_boxes, out_classes, num, out_scores)

        # EfficientDet
        if framework == "efficientdet":
            # get model input
            model_image_size = (512, 512)
            img_data, input_name = get_efficientdet_input(img, session, model_image_size)
            ori_imgs, framed_imgs, framed_metas = ef_preprocess(pth, max_size=512)

            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
            x = x.cpu()
            x = np.array(x)

            # predict with ONNX runtime
            start = time.time()
            for j in range(10):
                sess_out = session.run(None, {input_name: x})

                regression = sess_out[5]
                classification = sess_out[6]
                anchors = sess_out[7]

                # postprocess
                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()

                regression = torch.from_numpy(regression).float().to(device)
                classification = torch.from_numpy(classification).float().to(device)
                out = efficientdet_postprocess(x,
                                               anchors, regression, classification,
                                               regressBoxes, clipBoxes,
                                               threshold, iou_threshold)
                out = invert_affine(framed_metas, out)
            end = time.time()
            inference_time = round((end - start) / 10, 2)
            print("Inference time: ", inference_time, " sec")

            # display and save predictions
            efficientdet_display(out, ori_imgs, classes, num, imshow=False, imwrite=True)

        num += 1


def main():
    # pass in framework, .onnx model, and test images
    try:
        framework = sys.argv[1]
        model = sys.argv[2]
        images = sys.argv[3]
    except(Exception):
        print("Must provide framework, model, and images path.")

    print("Framework:", framework)
    print("Model:", model)
    print("Image:", images, "\n")

    # start onnxruntime session
    session = onnxruntime.InferenceSession(model)

    # run inferences
    inference(session, framework, images)


if __name__ == "__main__":
    main()
