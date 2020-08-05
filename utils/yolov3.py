# https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/yoloV3_object_detection_onnxruntime_inference.ipynb

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


# resize the test image to match the model
def resize_image(img, size):
    # Resize image with unchanged aspect ratio using padding
    img_w, img_h = img.size
    w, h = size
    scale = min(w / img_w, h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    image = img.resize((new_w, new_h), Image.BICUBIC)
    new_img = Image.new('RGB', size, (128, 128, 128))
    new_img.paste(image, ((w - new_w) // 2, (h - new_h) // 2))
    return new_img


# preprocess the image for testing
def yolov3_preprocess(img, model_image_size):
    resized_image = resize_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data


# get img_data, input_name, and input_shape
def get_yolov3_input(img, session, model_image_size):
    img_data = yolov3_preprocess(img, model_image_size)
    input_name = session.get_inputs()[0].name
    img_size = np.array([img.size[1], img.size[0]], dtype=np.float32).reshape(1, 2)
    return img_data, input_name, img_size


# postprocess the predictions
def yolov3_postprocess(boxes, scores, indices, CLASSES):
    objects_identified = indices.shape[1]
    out_boxes, out_scores, out_classes = [], [], []
    if objects_identified > 0:
        for idx_ in indices[0]:
            # print(idx_)
            # change this if using more than one class
            out_classes.append(CLASSES[0])
            out_scores.append(scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(boxes[idx_1])
        i = 0

    return out_boxes, out_scores, out_classes, objects_identified


# display predictions with bounding boxes
def yolov3_display_objdetect_image(image, out_boxes, out_classes, num, out_scores, objects_identified=None, save=True):
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    objects_identified = len(out_boxes)
    print(str(len(out_boxes)) + " objects identified in source image.")
    for i in range(objects_identified):
        y1, x1, y2, x2 = out_boxes[i]
        class_pred = out_classes[i]
        print(str(round(out_scores[i] * 100, 2)) + "% " + out_classes[i])
        color = 'blue'
        box_h = (y2 - y1)
        box_w = (x2 - x1)
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(bbox)
        plt.text(x1, y1, s=class_pred, color='white', verticalalignment='bottom', bbox={'color': color, 'pad': 0})

    plt.axis('off')
    # save image
    image_name = "data/predictions/" + "predictions" + str(num) + ".jpg"
    print("Saving prediction to: " + image_name)
    plt.savefig(image_name, bbox_inches='tight', pad_inches=0.0)
    if save:
        # uncomment this to show detection files
        # plt.show()
        pass
    else:
        img = plt.imread(image_name)
        os.remove(image_name)
        return img