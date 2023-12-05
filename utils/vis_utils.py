import cv2
import os

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.3
THICKNESS = 1

# Colors
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
PURPLE = (128, 0, 128)
GREY = (128, 128, 128)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
colour_lsts = [BLUE, PURPLE, GREY, GREEN]

pos_lsts = [1, 2, 3, 4]


def draw_label_gt(input_image, label, left, bottom):
    """Draw gt text onto image at location."""

    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]

    # The top-left corner of the background rectangle is calculated by subtracting the text dimension and baseline from the bottom left point.
    top_left_corner = (left, bottom - dim[1] - baseline)

    # Draw the text starting from bottom left, moving up by the height of the text.
    text_origin = (left, bottom - baseline)

    # Use text size to create a BLACK rectangle for background
    cv2.rectangle(input_image, top_left_corner, (left + dim[0], bottom), BLACK, cv2.FILLED)

    # Display text inside the rectangle.
    cv2.putText(input_image, label, text_origin, FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def draw_label_custom_ci(input_image, label, left, top, is_python_end):
    """Draw text onto image at location."""

    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]

    if is_python_end:
        # Draw the text in the top-left corner
        text_origin = (left, top + dim[1])
        # Use text size to create a BLACK rectangle for background
        cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
    else:
        # Draw the text in the bottom-left corner
        text_origin = (left, top - baseline)
        # Adjust the top coordinate for the rectangle to accommodate text height
        cv2.rectangle(input_image, (left, top - dim[1] - baseline), (left + dim[0], top), BLACK, cv2.FILLED)

    # Display text inside the rectangle.
    cv2.putText(input_image, label, text_origin, FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def draw_label_custom_cii(input_image, label, left, top, first_model=True):
    """Draw text onto image at location."""

    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]

    if first_model:
        # Draw the text in the top-left corner
        text_origin = (left, top + dim[1])
        # Use text size to create a BLACK rectangle for background
        cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
    else:
        # Draw the text in the bottom-left corner
        text_origin = (left, top - baseline)
        # Adjust the top coordinate for the rectangle to accommodate text height
        cv2.rectangle(input_image, (left, top - dim[1] - baseline), (left + dim[0], top), BLACK, cv2.FILLED)

    # Display text inside the rectangle.
    cv2.putText(input_image, label, text_origin, FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def display_image_ci(image_path, save_path, p_detections, c_detections, gt_infos, save_image=False, display_gt=True):
    # Load the image
    image = cv2.imread(image_path)

    for detection in c_detections:
        box = detection["box"]
        label = detection['label']
        confidence = detection['confidence']

        left, top, width, height = box  # extract the rectangle shape info
        cv2.rectangle(image, (left, top), (left + width, top + height), PURPLE, 3 * THICKNESS)
        label = "{}:{:.2f}".format(label, confidence)
        draw_label_custom_ci(image, label, left, top, is_python_end=False)

    for detection in p_detections:
        box = detection["box"]
        label = detection['label']
        confidence = detection['confidence']

        left, top, width, height = box  # extract the rectangle shape info
        cv2.rectangle(image, (left, top), (left + width, top + height), BLUE, THICKNESS)
        label = "{}:{:.2f}".format(label, confidence)
        draw_label_custom_ci(image, label, left, top, is_python_end=True)

    if display_gt:
        for gt_info in gt_infos:
            box = gt_info["bbox"]
            name = gt_info['category_name']

            left, top, width, height = [int(coord) for coord in box]
            cv2.rectangle(image, (left, top), (left + width, top + height), GREEN, 1 * THICKNESS)
            label = "{}".format(name)
            draw_label_gt(image, label, left, top + height)

    if save_image:
        cv2.imwrite(os.path.join(save_path, "infer-" + os.path.basename(image_path)), image)

    # Display the image
    cv2.imshow('Image', image)
    cv2.waitKey(3000)  # Display the window for seconds
    cv2.destroyAllWindows()


def display_image_cii(image_load_path, image_save_path, json_models, json_gts,
                              save_image=True, display_gt=True):
    # Load the image
    image = cv2.imread(image_load_path)

    for i, (model_name, json_detections) in enumerate(json_models.items()):

        frame_color = colour_lsts[i]  # choose distinct colour
        # label_pos = pos_lsts

        for detection in json_detections:
            box = detection["box"]
            label = detection['label']
            confidence = detection['confidence']

            left, top, width, height = box  # extract the rectangle shape info
            cv2.rectangle(image, (left, top), (left + width, top + height), frame_color, THICKNESS)
            label = "{}:{:.2f}".format(label, confidence)
            draw_label_custom_cii(image, label, left, top, first_model=i)  # use i to control the position of labels

    if display_gt:

        gt_frame_color = GREEN

        for gt_info in json_gts:
            box = gt_info["bbox"]
            name = gt_info['category_name']

            left, top, width, height = [int(coord) for coord in box]
            cv2.rectangle(image, (left, top), (left + width, top + height), gt_frame_color, THICKNESS)
            label = "{}".format(name)
            draw_label_gt(image, label, left, top + height)

    if save_image:
        cv2.imwrite(os.path.join(image_save_path, "infer-" + os.path.basename(image_load_path)), image)

    # Display the image
    cv2.imshow('Image', image)
    cv2.waitKey(3000)  # Display the window for seconds
    cv2.destroyAllWindows()
