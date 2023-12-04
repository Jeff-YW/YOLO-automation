from yolov5 import pre_process, post_process
import cv2
import sys
import json

if __name__ == '__main__':

    # Load class names.
    classesFile = "coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')


    frame = cv2.imread(sys.argv[1])

    modelWeights = sys.argv[2]
    net = cv2.dnn.readNet(modelWeights)

    # Process image.
    detections = pre_process(frame, net)
    img, results = post_process(frame.copy(), detections, classes=classes)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    inference_time = t * 1000.0 / cv2.getTickFrequency()
    label = 'Inference time: %.2f ms' % (inference_time)

    results.append({"inference time":inference_time})

    # std out the JSON format
    print(json.dumps(results))