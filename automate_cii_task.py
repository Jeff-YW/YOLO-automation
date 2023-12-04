import os.path
import subprocess
import json
import yaml
import datetime
from utils.vis_utils import display_image_cii
from utils.log_utils import record_gt, record_pred_cii
from utils.gen_utils import extract_id, read_text_file
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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

def coco_result_evaluation_cii():
    # Filter annotations to only include those for the subset of images
    coco.dataset['annotations'] = filtered_annotations
    coco.createIndex()

    for _, (model_name, pred_jsons) in enumerate(coco_json_models.items()):
        # prediction object loaded
        cocoDt = coco.loadRes(pred_jsons)

        # evaluation object created
        cocoEval = COCOeval(coco, cocoDt, 'bbox')

        # evaluate
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        print(f"Evaluated Model: {model_name}\n")
        print(cocoEval.stats)
        print("\n\n")

        # save to report
        if log_prediction_flag:
            with open(save_report_path, 'a') as f:  # Use 'a' for append mode
                f.write(f"Model Name: {model_name}\n")  # Write the model name
                f.write("Stats:\n")
                f.write(format_stats(cocoEval.stats))  # Write the stats
                f.write("\n\n")  # Add a newline for separation between entries
                f.write("\n")  # Add a newline for separation between entries


# get the current working dir
wrk_dir = os.getcwd()

# YAML file path
yaml_file_path = os.path.join(wrk_dir, 'things.yaml')

# load YAML params
params = load_yaml(yaml_file_path)
cii_params = params['cii_task']

# create the save folder if not already exists
abs_save_path = os.path.join(wrk_dir, cii_params["save_path"])
if not os.path.exists(abs_save_path):
    os.makedirs(abs_save_path)

# Extracting values from the YAML content
val2017_text_path = params['img_source_file']
annFilepath = params['ann_path']

nums_of_pics = cii_params['num_of_pic']  # number of pics defined

# flags for visualizing results in the report
create_vis_output_flag = cii_params['vis_pred']
vis_gt_flag = cii_params['vis_gt']
log_prediction_flag = cii_params['log_pred']
log_gt_flag = cii_params['log_gt']
save_img_flag = cii_params["save_image"]

# Models, Runfiles, and Save file paths can also be extracted similarly
print(f"number of YOLO models for evaluation: {len(cii_params['models'])}")

models = {} # save as a new dict
for i in range(len(cii_params['models'])):
    models[cii_params['models'][i]['name']] = cii_params['models'][i]['path']

runfiles_config = cii_params['runfiles']  # Python and Cplusplus (saved as dict in Python)
save_report_path = os.path.join(abs_save_path, cii_params['report_name'])

# Get the current date and time
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

if log_prediction_flag:
    # Open the file in 'w' mode, creating a new one or truncating if it exists
    with open(save_report_path, 'w') as new_file:
        new_file.write(f"File created on: {formatted_datetime}\n")

# calling the cocoAPI
coco = COCO(annFilepath)

# extract all the information from the coco paths
cats = coco.loadCats(coco.getCatIds())

# create a dict for mapping Annotation ids to categories' actual names
cat_names = {cat['id']: cat['name'] for cat in cats}
cat_ids = {cat['name']: cat['id'] for cat in cats}

val_image_paths = read_text_file(val2017_text_path)  # load the paths to the pictures

# create a dict to save the predictions of models
coco_json_models = {model['name']: [] for model in cii_params['models']}
filtered_annotations = []

for idx, val_image_path in enumerate(val_image_paths):
    if idx >= nums_of_pics:
        break

    img_path = f'{os.path.join("coco", val_image_path)}'

    ########### Extract GT INFO ################
    coco_image_id = int(extract_id(val_image_path))

    imgInfo = coco.loadImgs(coco_image_id)[0]

    # extract all the ground truth annotations ids
    annIds = coco.getAnnIds(imgIds=imgInfo['id'])

    # get all the annotation ids' info
    anns = coco.loadAnns(annIds)

    filtered_annotations.extend(anns)

    ######## PREPARE GT LOG #########
    if create_vis_output_flag or log_gt_flag:

        coco_gt_infos = []

        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])  # convert the bbox to integer for opencv post-processing

            # extract the coco ID of the class that the annotation belongs to
            category_id = ann['category_id']

            # convert the coco ID to actual class name
            category_name = cat_names.get(category_id, "Unknown")

            coco_gt_infos.append({"bbox": [x, y, w, h], "category_name": category_name})

    json_detection_models = {}
    json_inference_times = {}

    # run subprocess on models ...
    for i, (model_name, model_path) in enumerate(models.items()):
        # call python opencv YOLO prediction
        output_model_yolo = subprocess.run(['python', runfiles_config['path'], img_path, model_path],
                                           stdout=subprocess.PIPE)

        # decode the Json ostream from the sub-process calling
        json_model_yolo = json.loads(output_model_yolo.stdout.decode('utf-8'))

        # Extracting all detections (assuming all but last are detections)
        json_python_pred = json_model_yolo[:-1]  # list
        # Extracting the inference time (assuming it is the last entry)
        json_python_inference_time = json_model_yolo[-1]

        # add to the temp dict ...
        json_detection_models[model_name] = json_python_pred
        json_inference_times[model_name] = json_python_inference_time['inference time']

        # inference_times.append(json_python_inference_time)

        # extract the gt image coco info ... using COCO JSON FORMAT!
        for _, json_py in enumerate(json_python_pred):
            coco_json_models[model_name].append({
                'image_id': coco_image_id,
                'category_id': cat_ids[json_py['label']],
                'bbox': [float(json_py["box"][0]), float(json_py["box"][1]),
                         float(json_py["box"][2]), float(json_py["box"][3])],
                'score': float(json_py['confidence']),
                'area': float((json_py["box"][2]) * (json_py["box"][3]))
            })

    ######## VISUALIZING #########
    if create_vis_output_flag:

        display_image_cii(img_path, abs_save_path, json_detection_models, coco_gt_infos,
                                  save_image=save_img_flag, display_gt=vis_gt_flag)

    ######### LOGGING ############
    if log_prediction_flag:
        record_pred_cii(val_image_path, save_report_path, json_detection_models,
                    json_inference_times)

        if log_gt_flag:
            record_gt(val_image_path, save_report_path, coco_gt_infos)


coco_result_evaluation_cii()