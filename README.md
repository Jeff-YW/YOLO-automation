
# YOLO-automation

This repository contains tools and scripts for automating the comparison of different aspects of `YOLO` implementations and models. 
It is designed to facilitate the evaluation of `C++` and `Python`` implementation equivalence and to compare the performance and inference time of different YOLO models.

## Getting Started

Clone the repository and `cd` into the directory to begin:

```bash
git clone [URL_TO_THIS_REPO]
cd [LOCAL_PATH_TO_THIS_REPO]
```

## Repository Structure

- `utils/`: Contains utility scripts and modules to support the main tasks.
- `models/`: Includes ONNX YOLO models such as YOLO5s, YOLO5n and YOLO5m used for comparisons.
- `json/`: Stores JSON cpp header files.
- `.gitignore`: Configured to ignore COCO dataset folders and the result folders and IDE configuration files.
- `CMakeLists.txt`: CMake configuration for building C++ files.
- `Convert_PyTorch_models.ipynb`: Jupyter notebook to convert PyTorch models.
- `automate_ci_task.py`: Main script to automate task (ci), checking implementation equivalence.
- `automate_cii_task.py`: Main script to automate task (cii), comparing model performance.
- `things.yaml`: A structured files used to control the configurations, display and directories to folder, help to run Python files from the current working directory.
- `load_zip_file.py`: A useful file to help unzip the downloaded coco dataset into the current working directory. 
- Other supportive scripts and configuration files.

## Main Scripts

- `automate_ci_task.py`: This script is used to validate the equivalence of C++ and Python YOLO outputs (post NMS).
- `automate_cii_task.py`: Facilitates the comparison between YOLO5n and YOLO5m models, in terms of detection performance and inference time.

## Get COCO Datasets

The dataset from the COCO 2017 Object Detection Task. The coco datasets needs to be downloaded in advance since the tasks rely on the dataset, it is for the best to download and unzip the dataset into the current working directory. If not, there is a way to link to the coco dataset by 
changing the path to dataset. As shown below, by changing the two parameters will enable the python script to track down the images' locations locally.

```YAML
img_source_file: "coco/val2017.txt"
ann_path:  "coco/annotations/instances_val2017.json"
```

## Results

The results are saved as text files within an automatically generated directory. Additionally, results are available in terminal print-out format and visualized prediction results (i.e., picture results).

## Configuration

The YAML file (`things.yaml`) contains important parameters to control:

- Enables switching between any of the three available YOLO models for task ci.
- Allows the comparison of either two of the three models for object detection task performance evaluation in task cii.

To enable print-out results and photo displayed results, make the necessary changes to the YAML file parameters.

## Requirements
The python packages required are:

`PyYAML`

`opencv-python`

`pycocotools`

You can run the following scripts to install the necessary packages. 
```
pip install -r requirements.txt
```

## Building C++ Files

C++ Files need to be built first! before carrying out the automation task and the directory name should be consistent as in the `things.yaml` so that the python automating script can find the correct file to run!

Instructions for building the C++ files with CMake:

Linux
```
mkdir build
cd build
cmake ..
cmake --build .
```

Windows
```
rmdir /s /q build
cmake -S . -B build
cmake --build build --config Release
```

If you want to run the C++ code, make sure you `cd` to the current working directory (not the build directory), and use the following command:

Linux
```
./build/Med_TA sample.jpg model/yolov5s.onnx
```

Windows
```
.\build\Release\Med_TA.exe sample.jpg model\yolov5s.onnx
```

## Automate Task c(i)

Simply run

```
python automate_ci_task.py
```

## Automate Task c(ii)

Simply run

```
python automate_cii_task.py
```

## Acknowledgments

Thanks for this opportunity to demonstrate my understanding and skills! Through completing this challange, I have had the opportunity to appreciate the hands-on tasks at your group and overall I enjoy working on it. Looking forward to receiving further notice! 
