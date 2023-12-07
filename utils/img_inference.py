from yolov5 import pre_process, post_process
import cv2
import sys
import json
import os
import glob
from utils.gen_utils import extract_id, data_loader



def inference(folder_path, model_path, nums_of_pic):
    # Load class names.
    classesFile = "coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    net = cv2.dnn.readNet(model_path)

    inferences = []

    image_loader = data_loader(folder_path, nums_of_pic)

    for _ in range(nums_of_pic):
        img_path = next(image_loader)

        frame = cv2.imread(img_path)

        # Process image.
        detections = pre_process(frame, net)
        img, results = post_process(frame.copy(), detections, classes=classes)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        inference_time = t * 1000.0 / cv2.getTickFrequency()
        label = 'Inference time: %.2f ms' % (inference_time)

        results.append({"inference time": inference_time})

        results.append({'id': extract_id(os.path.basename(img_path))})

        inferences.append(results)

    # json_inferences = json.dumps(inferences)

    return inferences


if __name__ == '__main__':
    model_weights = "models/yolov5s.onnx"

    folder_path = "../coco/images/val2017"

    num_pics = 20

    inferenced_results = inference(folder_path, model_weights, num_pics)

    for item in inferenced_results:
        print(item[-2]["inference time"])
