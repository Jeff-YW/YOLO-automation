import os.path
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.img_inference import inference
import datetime
from utils.analyze_results import analyze_cii


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
        print(f"Evaluated Model: {model_name}\n")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        print(cocoEval.stats)
        print(f"Average Inference Time ({model_name}): {sum(efficiency_models[model_name])/len(efficiency_models[model_name])}\n")
        print("\n\n")

        # save to report
        if log_prediction_flag:
            with open(save_report_path, 'a') as f:  # Use 'a' for append mode
                f.write(f"Model Name: {model_name}\n")  # Write the model name
                f.write(f"Average Inference Time ({model_name}): {sum(efficiency_models[model_name])/len(efficiency_models[model_name])}\n")
                f.write("Stats:\n")
                f.write(format_stats(cocoEval.stats))  # Write the stats
                f.write("\n\n")  # Add a newline for separation between entries

    return


if __name__ == '__main__':

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
    val2017_img_folder_path = params['img_folder']

    nums_of_pics = cii_params['num_of_pic']  # number of pics defined

    # flags for visualizing results in the report
    create_vis_output_flag = cii_params['vis_pred']
    vis_gt_flag = cii_params['vis_gt']
    log_prediction_flag = cii_params['log_pred']
    log_gt_flag = cii_params['log_gt']
    save_img_flag = cii_params["save_image"]
    terminal_print_flag = cii_params["terminal_pred"]
    terminal_print_gt_flag = cii_params["terminal_gt"]

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
            new_file.write(f"----------- task c (ii) report -----------")
            new_file.write(f"File created on: {formatted_datetime}\n")
            for i in range(len(cii_params['models'])):
                new_file.write(f"Evaluation of Model: {models[cii_params['models'][i]['name']]}\n\n")
            new_file.write(f"Number of Pictures Analyzed: {nums_of_pics}\n\n")

    # calling the cocoAPI
    coco = COCO(annFilepath)

    # extract all the information from the coco paths
    cats = coco.loadCats(coco.getCatIds())

    # create a dict for mapping Annotation ids to categories' actual names
    cat_names = {cat['id']: cat['name'] for cat in cats}
    cat_ids = {cat['name']: cat['id'] for cat in cats}

    # # create a dict to save the predictions of models
    # coco_json_models = {model['name']: [] for model in cii_params['models']}
    # efficiency_models = {model['name']: [] for model in cii_params['models']}
    # filtered_annotations = []

    if terminal_print_flag:
        print("\n\n")
        print(f"----------- task c (i) report -----------\n")
        print(f"File created on: {formatted_datetime}\n")
        for i in range(len(cii_params['models'])):
            print(f"Evaluation of Model: {models[cii_params['models'][i]['name']]}\n\n")
        print(f"Number of Pictures Analyzed: {nums_of_pics}\n\n")


    results_python_yolos = {}

    for _, (model_name, model_path) in enumerate(models.items()):
        # call python opencv YOLO prediction
        results_python_yolos[model_name] =(inference(val2017_img_folder_path, model_path, nums_of_pics))


    # analyze results
    coco_json_models, efficiency_models, filtered_annotations = analyze_cii(nums_of_pics, coco, cat_names,
                                                                                  cat_ids,
                                                                                  results_python_yolos,
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

    coco_result_evaluation_cii()