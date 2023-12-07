import os.path
import subprocess
import json
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.img_inference import inference
from utils.analyze_results import analyze_ci
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
    print("Python Inference Summary\n")
    cocoEval_py.evaluate()
    cocoEval_py.accumulate()
    cocoEval_py.summarize()
    print(cocoEval_py.stats)
    print(f"Average Inference Time (Python): {sum(efficiency_py) / len(efficiency_py)}\n")

    ########################################################################

    # prediction object loaded CPP
    cocoDt_cpp = coco.loadRes(coco_json_py)

    # evaluation object created
    cocoEval_cpp = COCOeval(coco, cocoDt_cpp, 'bbox')

    # cocoeval to test the model performance
    print("Cpp Inference Summary\n")
    cocoEval_cpp.evaluate()
    cocoEval_cpp.accumulate()
    cocoEval_cpp.summarize()
    print(cocoEval_cpp.stats)
    print(f"Average Inference Time (Cpp): {sum(efficiency_cpp) / len(efficiency_cpp)}\n")

    # save to report
    if log_prediction_flag:
        with open(save_report_path, 'a') as f:  # Use 'a' for append mode
            f.write(f"On: Python\n")  # Write the model name
            f.write("Stats:\n")
            f.write(format_stats(cocoEval_py.stats))  # Write the stats
            f.write("\n\n")  # Add a newline for separation between entries
            f.write(f"Average Inference Time (Python): {sum(efficiency_py) / len(efficiency_py)}\n")
            f.write(f"On: Cpp\n")  # Write the model name
            f.write("Stats:\n")
            f.write(format_stats(cocoEval_cpp.stats))  # Write the stats
            f.write("\n\n")  # Add a newline for separation between entries
            f.write(f"Average Inference Time (Cpp): {sum(efficiency_cpp) / len(efficiency_cpp)}\n")

    return


if __name__ == '__main__':

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
    val2017_img_folder_path = params['img_folder']

    nums_of_pics = ci_params['num_of_pic']  # number of pics defined

    # flags for visualizing results in the report
    create_vis_output_flag = ci_params['vis_pred']
    vis_gt_flag = ci_params['vis_gt']
    log_prediction_flag = ci_params['log_pred']
    log_gt_flag = ci_params['log_gt']
    save_img_flag = ci_params["save_image"]
    terminal_print_flag = ci_params["terminal_pred"]
    terminal_print_gt_flag = ci_params["terminal_gt"]

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
            new_file.write(f"----------- task c (i) report -----------\n")
            new_file.write(f"File created on: {formatted_datetime}\n")
            new_file.write(f"Evaluation of Model: {model_name}\n\n")
            new_file.write(f"Number of Pictures Analyzed: {nums_of_pics}\n\n")

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

    if terminal_print_flag:
        print("\n\n")
        print(f"----------- task c (i) report -----------")
        print(f"File created on: {formatted_datetime}\n")
        print(f"Evaluation of Model: {model_name}\n\n")
        print(f"Number of Pictures Analyzed: {nums_of_pics}\n\n")


    # call python opencv YOLO prediction
    results_python_yolo = inference(val2017_img_folder_path, model_path, nums_of_pics)

    # call C plusplus opencv YOLO prediction
    results_c_yolo = subprocess.run([runfiles_config['cplusplus_path'], val2017_img_folder_path,
                                     model_path, str(nums_of_pics)], stdout=subprocess.PIPE)

    results_c_yolo = json.loads(results_c_yolo.stdout.decode('utf-8'))

    # analyze results
    coco_json_py, coco_json_cpp, filtered_annotations, efficiency_py, efficiency_cpp = analyze_ci(coco, cat_names,
                                                                                                  cat_ids,
                                                                                                  results_python_yolo,
                                                                                                  results_c_yolo,
                                                                                                  val2017_img_folder_path,
                                                                                                  abs_save_path,
                                                                                                  save_report_path,
                                                                                                  vis_img_f=create_vis_output_flag,
                                                                                                  log_img_f = log_prediction_flag,
                                                                                                  print_img_f =terminal_print_flag,
                                                                                                  vis_gt_f =vis_gt_flag,
                                                                                                  log_gt_f=log_gt_flag,
                                                                                                  print_gt_f=terminal_print_gt_flag,
                                                                                                  save_image_f=save_img_flag)

    coco_result_evaluation_ci()
