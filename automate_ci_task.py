import os.path
import subprocess
import json
import yaml
from utils.vis_utils import display_image_ci
from utils.log_utils import record_gt, record_pred_ci
from utils.gen_utils import extract_id, read_text_file
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import datetime


# Load YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def format_stats(stats):
    labels = [
        "Average Precision (AP) @ [ IoU=0.50:0.95 | area= all | maxDets=100 ]",
        "Average Precision (AP) @ [ IoU=0.50 | area= all | maxDets=100 ]",
        "Average Precision (AP) @ [ IoU=0.75 | area= all | maxDets=100 ]",
        "Average Precision (AP) @ [ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "Average Precision (AP) @ [ IoU=0.50:0.95 | area= medium | maxDets=100 ]",
        "Average Precision (AP) @ [ IoU=0.50:0.95 | area= large | maxDets=100 ]",
        "Average Recall (AR) @ [ IoU=0.50:0.95 | area= all | maxDets= 1 ]",
        "Average Recall (AR) @ [ IoU=0.50:0.95 | area= all | maxDets= 10 ]",
        "Average Recall (AR) @ [ IoU=0.50:0.95 | area= all | maxDets=100 ]",
        "Average Recall (AR) @ [ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "Average Recall (AR) @ [ IoU=0.50:0.95 | area= medium | maxDets=100 ]",
        "Average Recall (AR) @ [ IoU=0.50:0.95 | area= large | maxDets=100 ]"
    ]
    formatted_stats = "\n".join(f"{label}: {stat:.3f}" for label, stat in zip(labels, stats))
    return formatted_stats


def coco_result_evaluation_ci():
    # Filter annotations to only include those for the subset of images
    coco.dataset['annotations'] = filtered_annotations
    coco.createIndex()

    # prediction object loaded PYTHON
    cocoDt_py = coco.loadRes(coco_json_py)

    # evaluation object created
    cocoEval_py = COCOeval(coco, cocoDt_py, 'bbox')

    # cocoeval to test the model performance
    cocoEval_py.evaluate()
    cocoEval_py.accumulate()
    cocoEval_py.summarize()
    print(cocoEval_py.stats)

    ##################

    # prediction object loaded CPP
    cocoDt_cpp = coco.loadRes(coco_json_py)

    # evaluation object created
    cocoEval_cpp = COCOeval(coco, cocoDt_cpp, 'bbox')

    # cocoeval to test the model performance
    cocoEval_cpp.evaluate()
    cocoEval_cpp.accumulate()
    cocoEval_cpp.summarize()
    print(cocoEval_cpp.stats)

    # save to report
    if log_prediction_flag:
        with open(save_report_path, 'a') as f:  # Use 'a' for append mode
            f.write(f"On: Python\n")  # Write the model name
            f.write("Stats:\n")
            f.write(format_stats(cocoEval_py.stats))  # Write the stats
            f.write("\n\n")  # Add a newline for separation between entries
            f.write(f"On: Cpp\n")  # Write the model name
            f.write("Stats:\n")
            f.write(format_stats(cocoEval_cpp.stats))  # Write the stats
            f.write("\n\n")  # Add a newline for separation between entries


# get the current working dir
wrk_dir = os.getcwd()

# YAML file path
yaml_file_path = os.path.join(wrk_dir, 'things.yaml')

# load YAML params
params = load_yaml(yaml_file_path)
ci_params = params['ci_task']

# create the save folder if not already exists
abs_save_path = os.path.join(wrk_dir, ci_params["save_path"])
if not os.path.exists(abs_save_path):
    os.makedirs(abs_save_path)

# Extracting values from the YAML content
val2017_text_path = params['img_source_file']
annFilepath = params['ann_path']

nums_of_pics = ci_params['num_of_pic']  # number of pics defined

# flags for visualizing results in the report
create_vis_output_flag = ci_params['vis_pred']
vis_gt_flag = ci_params['vis_gt']
log_prediction_flag = ci_params['log_pred']
log_gt_flag = ci_params['log_gt']
save_img_flag = ci_params["save_image"]

# Models, Runfiles, and Save file paths can also be extracted similarly
model_name = ci_params['models'][0]['name']
model_path = ci_params['models'][0]['path']  # for task ci, only 1 model
runfiles_config = ci_params['runfiles']  # Python and Cplusplus (saved as dict in Python)
save_report_path = os.path.join(abs_save_path, ci_params['report_name'])

# Get the current date and time
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

if log_prediction_flag:
    # Open the file in 'w' mode, creating a new one or truncating if it exists
    with open(save_report_path, 'w') as new_file:
        new_file.write(f"File created on: {formatted_datetime}\n")

print(f"evaluating model: {model_name}")

# load the gt names of the classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# calling the cocoAPI
coco = COCO(annFilepath)

# extract all the information from the coco paths
cats = coco.loadCats(coco.getCatIds())

# create a dict for mapping Annotation ids to categories actual name
cat_names = {cat['id']: cat['name'] for cat in cats}
cat_ids = {cat['name']: cat['id'] for cat in cats}

val_image_paths = read_text_file(val2017_text_path)  # load the paths to the pictures

coco_json_py = []
coco_json_cpp = []
filtered_annotations = []

for idx, val_image_path in enumerate(val_image_paths):

    # print(f"{val_image_path}")

    if idx >= nums_of_pics:
        break

    img_path = f'{os.path.join("coco", val_image_path)}'

    # call python opencv YOLO prediction
    result_python_yolo = subprocess.run(['python', runfiles_config['python_path'], img_path],
                                        stdout=subprocess.PIPE)
    # call C plusplus opencv YOLO prediction
    result_c_yolo = subprocess.run([runfiles_config['cplusplus_path'], img_path], stdout=subprocess.PIPE)

    # decode the Json ostream from the sub-process calling
    json_python = json.loads(result_python_yolo.stdout.decode('utf-8'))
    json_c_plusplus = json.loads(result_c_yolo.stdout.decode('utf-8'))

    # Extracting all detections (assuming all but last are detections)
    json_python_pred = json_python[:-1]  # list
    # Extracting the inference time (assuming it is the last entry)
    json_python_inference_time = json_python[-1]

    json_c_plusplus_pred = json_c_plusplus[:-1]
    json_c_plusplus_inference_time = json_c_plusplus[-1]

    ########### Extract GT INFO ################
    coco_image_id = int(extract_id(val_image_path))

    imgInfo = coco.loadImgs(coco_image_id)[0]

    # extract all the ground truth annotations ids
    annIds = coco.getAnnIds(imgIds=imgInfo['id'])

    # get all the annotation ids' info
    anns = coco.loadAnns(annIds)

    filtered_annotations.extend(anns)

    if create_vis_output_flag or log_gt_flag:

        coco_gt_infos = []

        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])  # convert the bbox to integer for opencv post processing

            # extract the coco ID of the class that the annotation belongs to
            category_id = ann['category_id']

            # convert the coco ID to actual class name
            category_name = cat_names.get(category_id, "Unknown")

            coco_gt_infos.append({"bbox": [x, y, w, h], "category_name": category_name})

    if create_vis_output_flag:
        # draw the prediction boxes on the gt picture and save here
        display_image_ci(img_path, abs_save_path, json_python_pred, json_c_plusplus_pred, coco_gt_infos,
                         save_image=save_img_flag, display_gt=vis_gt_flag)

    if log_prediction_flag:
        record_pred_ci(val_image_path, save_report_path, json_python_pred, json_c_plusplus_pred,
                       json_python_inference_time,
                       json_c_plusplus_inference_time)

        if log_gt_flag:
            record_gt(val_image_path, save_report_path, coco_gt_infos)

    # extract the gt image coco info ...

    for _, json_py in enumerate(json_python_pred):
        coco_json_py.append({
            'image_id': coco_image_id,
            'category_id': cat_ids[json_py['label']],
            'bbox': [float(json_py["box"][0]), float(json_py["box"][1]),
                     float(json_py["box"][2]), float(json_py["box"][3])],
            'score': float(json_py['confidence']),
            'area': float((json_py["box"][2]) * (json_py["box"][3]))
        })

    for _, json_cpp in enumerate(json_c_plusplus_pred):
        coco_json_cpp.append({
            'image_id': coco_image_id,
            'category_id': cat_ids[json_cpp['label']],
            'bbox': [float(json_cpp["box"][0]), float(json_cpp["box"][1]),
                     float(json_cpp["box"][2] - json_cpp["box"][0]),
                     float(json_cpp["box"][3] - json_cpp["box"][1])],
            'score': float(json_cpp['confidence']),
            'area': float((json_cpp["box"][2] - json_cpp["box"][0]) * (json_cpp["box"][3] - json_cpp["box"][1]))
        })

coco_result_evaluation_ci()
