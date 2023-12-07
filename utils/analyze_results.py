from utils.vis_utils import display_image_ci, display_image_cii
from utils.log_utils import record_gt, record_pred_ci, print_gt, print_pred_ci, record_pred_cii, print_pred_cii
from utils.gen_utils import extract_id, data_loader
import os


def analyze_cii(N, coco_API, coco_cat_names, coco_cat_ids, results_models,
                img_folder_path, img_save_path, save_report_path,
                vis_img_f=True,
                log_img_f=True,
                print_img_f=True,
                vis_gt_f=True,
                log_gt_f=True,
                print_gt_f=True,
                save_image_f=True):
    num_of_models = len(results_models)

    coco_json_models = {model_name: [] for model_name, jsons in results_models.items()}
    efficiency_models = {model_name: [] for model_name, jsons in results_models.items()}
    filtered_annotations = []

    img_loader = data_loader(img_folder_path, N)

    for i in range(N):

        ########### Extract GT INFO ################
        curr_img_path = next(img_loader)

        coco_image_id = int(extract_id(os.path.basename(curr_img_path)))

        imgInfo = coco_API.loadImgs(coco_image_id)[0]

        # extract all the ground truth annotations ids
        annIds = coco_API.getAnnIds(imgIds=imgInfo['id'])

        # get all the annotation ids' info
        anns = coco_API.loadAnns(annIds)

        filtered_annotations.extend(anns)

        # if create_vis_output_flag or log_gt_flag or terminal_print_flag:

        # if log_gt:

        coco_gt_infos = []

        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])  # convert the bbox to integer for opencv post processing

            # extract the coco ID of the class that the annotation belongs to
            category_id = ann['category_id']

            # convert the coco ID to actual class name
            category_name = coco_cat_names.get(category_id, "Unknown")

            coco_gt_infos.append({"bbox": [x, y, w, h], "category_name": category_name})

        json_detection_models = {}
        json_inference_times = {}

        for _, (model_name, model_json_infer) in enumerate(results_models.items()):
            json_detection_models[model_name] = model_json_infer[i][:-2]
            json_inference_times[model_name] = model_json_infer[i][-2]
            efficiency_models[model_name].append(float(model_json_infer[i][-2]['inference time']))

            # extract the gt image coco info ... using COCO JSON FORMAT!
            for _, json_py in enumerate(json_detection_models[model_name]):
                coco_json_models[model_name].append({
                    'image_id': coco_image_id,
                    'category_id': coco_cat_ids[json_py['label']],
                    'bbox': [float(json_py["box"][0]), float(json_py["box"][1]),
                             float(json_py["box"][2]), float(json_py["box"][3])],
                    'score': float(json_py['confidence']),
                    'area': float((json_py["box"][2]) * (json_py["box"][3]))
                })

        ###################################################################################################
        if vis_img_f:
            pass
            # draw the prediction boxes on the gt picture and save here
            display_image_cii(curr_img_path, img_save_path, json_detection_models, coco_gt_infos,
                              save_image=save_image_f, display_gt=vis_img_f)

        if log_img_f:
            pass
            record_pred_cii(os.path.basename(curr_img_path), save_report_path, json_detection_models,
                            json_inference_times)

            if log_gt_f:
                pass
                record_gt(curr_img_path, save_report_path, coco_gt_infos)

        if print_img_f:
            pass
            print_pred_cii(os.path.basename(curr_img_path), json_detection_models,
                           json_inference_times)

            if print_gt_f:
                pass
                print_gt(curr_img_path, coco_gt_infos)

    return coco_json_models, efficiency_models, filtered_annotations


def analyze_ci(coco_API, coco_cat_names, coco_cat_ids, results_py, results_cpp,
               img_folder_path, img_save_path, save_report_path,
               vis_img_f=True,
               log_img_f=True,
               print_img_f=True,
               vis_gt_f=True,
               log_gt_f=True,
               print_gt_f=True,
               save_image_f=True):
    N = len(results_py)

    coco_json_py, coco_json_cpp, filtered_annotations = [], [], []
    efficiency_py = []
    efficiency_cpp = []

    img_loader = data_loader(img_folder_path, N)

    for _, (sub_lst_py, sub_lst_cpp) in enumerate(zip(results_py, results_cpp)):

        infer_result_py = sub_lst_py[:-2]
        infer_time_py = sub_lst_py[-2]
        img_id_py = sub_lst_py[-1]

        infer_result_cpp = sub_lst_cpp[:-2]
        infer_time_cpp = sub_lst_cpp[-2]
        img_id_cpp = sub_lst_cpp[-1]

        efficiency_py.append(float(infer_time_py["inference time"]))
        efficiency_cpp.append(float(infer_time_cpp["inference time"]))

        # if log_gt:
        ########### Extract GT INFO ################

        curr_img_path = next(img_loader)

        coco_image_id = int(extract_id(os.path.basename(curr_img_path)))

        imgInfo = coco_API.loadImgs(coco_image_id)[0]

        # extract all the ground truth annotations ids
        annIds = coco_API.getAnnIds(imgIds=imgInfo['id'])

        # get all the annotation ids' info
        anns = coco_API.loadAnns(annIds)

        filtered_annotations.extend(anns)

        # if create_vis_output_flag or log_gt_flag or terminal_print_flag:

        # if log_gt:

        coco_gt_infos = []

        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])  # convert the bbox to integer for opencv post processing

            # extract the coco ID of the class that the annotation belongs to
            category_id = ann['category_id']

            # convert the coco ID to actual class name
            category_name = coco_cat_names.get(category_id, "Unknown")

            coco_gt_infos.append({"bbox": [x, y, w, h], "category_name": category_name})

        ###################################################################################################
        if vis_img_f:
            pass
            # draw the prediction boxes on the gt picture and save here
            display_image_ci(curr_img_path, img_save_path, infer_result_py, infer_result_cpp, coco_gt_infos,
                             save_image_f, vis_gt_f)

        if log_img_f:
            pass
            record_pred_ci(os.path.basename(curr_img_path), save_report_path, infer_result_py, infer_result_cpp,
                           infer_time_py,
                           infer_time_cpp)

            if log_gt_f:
                pass
                record_gt(curr_img_path, save_report_path, coco_gt_infos)

        if print_img_f:
            pass
            print_pred_ci(os.path.basename(curr_img_path), infer_result_py, infer_result_cpp,
                          infer_time_py,
                          infer_time_cpp)

            if print_gt_f:
                pass
                print_gt(curr_img_path, coco_gt_infos)

        #######################################################################################################
        # extract the prediction of image coco info as COCO JSON format...
        for _, json_py in enumerate(infer_result_py):
            coco_json_py.append({
                'image_id': coco_image_id,
                'category_id': coco_cat_ids[json_py['label']],
                'bbox': [float(json_py["box"][0]), float(json_py["box"][1]),
                         float(json_py["box"][2]), float(json_py["box"][3])],
                'score': float(json_py['confidence']),
                'area': float((json_py["box"][2]) * (json_py["box"][3]))
            })

        for _, json_cpp in enumerate(infer_result_cpp):
            coco_json_cpp.append({
                'image_id': coco_image_id,
                'category_id': coco_cat_ids[json_cpp['label']],
                'bbox': [float(json_cpp["box"][0]), float(json_cpp["box"][1]),
                         float(json_cpp["box"][2] - json_cpp["box"][0]),
                         float(json_cpp["box"][3] - json_cpp["box"][1])],
                'score': float(json_cpp['confidence']),
                'area': float((json_cpp["box"][2] - json_cpp["box"][0]) * (json_cpp["box"][3] - json_cpp["box"][1]))
            })

    return coco_json_py, coco_json_cpp, filtered_annotations, efficiency_py, efficiency_cpp
