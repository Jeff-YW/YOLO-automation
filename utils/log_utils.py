from utils.gen_utils import extract_id


def append_to_file(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text + '\n')


def print_gt(image_path, gt_infos):
    print(f"Image Path: {image_path}")
    print(f"Number of Objects (Ground Truth): {len(gt_infos)}\n")

    print("{:<20} {:<30}".format("Object Class", "Bounding Box"))
    print("-" * 50)
    for gt_info in gt_infos:
        print("{:<20} {:<30}".format(gt_info['category_name'], str(gt_info['bbox'])))

    print("\n")


def print_pred_ci(image_path, p_detections, c_detections, p_time, c_time):
    img_id = extract_id(image_path)

    print(f"Image ID: {img_id}")
    print(f"Inference Time (Python-YOLO): {p_time['inference time']} s")
    print(f"Number of Objects Detected (Python-YOLO): {len(p_detections)}\n")

    print("{:<10} {:<15} {:<30}".format("Object", "Confidence", "Bounding Box"))
    print("-" * 55)
    for p_detection in p_detections:
        print("{:<10} {:<15} {:<30}".format(p_detection['label'], str(p_detection['confidence']),
                                            str(p_detection['box'])))

    print(f"\nInference Time (C++-YOLO): {c_time['inference time']} s")
    print(f"Number of Objects Detected (C++-YOLO): {len(c_detections)}\n")

    print("{:<10} {:<15} {:<30}".format("Object", "Confidence", "Bounding Box"))
    print("-" * 55)
    for c_detection in c_detections:
        print("{:<10} {:<15} {:<30}".format(c_detection['label'], str(c_detection['confidence']),
                                            str(c_detection['box'])))

    print("\n")


def print_pred_cii(image_path, detections_dict, inf_times_dict):
    img_id = extract_id(image_path)

    print(f"Image ID: {img_id}\n")

    for model_name, detections in detections_dict.items():
        inference_time = inf_times_dict[model_name]

        print(f"Inference Time ({model_name}): {inference_time} s")
        print(f"Number of Objects Detected ({model_name}): {len(detections)}\n")

        print("{:<10} {:<15} {:<30}".format("Object", "Confidence", "Bounding Box"))
        print("-" * 55)
        for detection in detections:
            print(
                "{:<10} {:<15} {:<30}".format(detection['label'], str(detection['confidence']), str(detection['box'])))
        print("\n")


def record_gt(image_path, file_path, gt_infos):
    append_to_file(file_path, f"Number of object (Ground truth): {len(gt_infos)}")

    for gt_info in gt_infos:
        append_to_file(file_path, f"Object Class (Ground truth): {gt_info['category_name']}")
        append_to_file(file_path, f"Bounding Box (Ground truth): {gt_info['bbox']}")


def record_pred_ci(image_path, file_path, p_detections, c_detections, p_time, c_time):
    img_id = extract_id(image_path)

    append_to_file(file_path, f"image id: {img_id}")

    append_to_file(file_path, f"inference time (Python-YOLO): {p_time['inference time']}")

    append_to_file(file_path, f"Number of object detected (Python-YOLO): {len(p_detections)}")

    for _, p_detection in enumerate(p_detections):
        append_to_file(file_path, f"Object: {p_detection['label']}")
        append_to_file(file_path, f"Confidence: {p_detection['confidence']}")
        append_to_file(file_path, f"Bounding box: {p_detection['box']}")

    append_to_file(file_path, f"inference time (Cplusplus-YOLO): {c_time['inference time']}")

    append_to_file(file_path, f"Number of object detected (Cplusplus-YOLO): {len(c_detections)}")

    for _, c_detection in enumerate(c_detections):
        append_to_file(file_path, f"Object: {c_detection['label']}")
        append_to_file(file_path, f"Confidence: {c_detection['confidence']}")
        append_to_file(file_path, f"Bounding box: {c_detection['box']}")


def record_pred_cii(image_path, file_path, detections_dict, inf_times_dict):
    img_id = extract_id(image_path)

    append_to_file(file_path, f"image id: {img_id}")

    for idx, ((model_name, detections), (_, time)) in enumerate(zip(detections_dict.items(), inf_times_dict.items())):

        append_to_file(file_path, f"inference time ({model_name}): {time}")
        append_to_file(file_path, f"Number of object detected ({model_name}): {len(detections)}")

        for _, detection in enumerate(detections):
            append_to_file(file_path, f"Object: {detection['label']}")
            append_to_file(file_path, f"Confidence: {detection['confidence']}")
            append_to_file(file_path, f"Bounding box: {detection['box']}")
