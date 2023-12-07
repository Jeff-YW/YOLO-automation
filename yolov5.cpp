// Include Libraries.
#include <opencv2/opencv.hpp>
#include <fstream>
#include "json/json.h"
#include <utility>
#include <filesystem>

namespace fs = std::filesystem;
// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);

// 创建 JSON 对象
Json::Value result(Json::arrayValue);


// Draw the predicted bounding box.
void draw_label(Mat& input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw black rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}


vector<Mat> pre_process(Mat &input_image, Net &net)
{
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}


Mat post_process(Mat &&input_image, vector<Mat> &outputs, const vector<string> &class_name)
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes; 

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) 
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD) 
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD) 
            {
                // Store class ID and confidence in the pre-defined respective vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += 85;
    }




    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++) 
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);


        // create a Json object
        Json::Value detection(Json::objectValue);
        detection["label"] = class_name[class_ids[idx]];
        detection["confidence"] = confidences[idx];
        detection["box"] = Json::Value(Json::arrayValue);
        detection["box"].append(box.x);
        detection["box"].append(box.y);
        detection["box"].append(box.width);
        detection["box"].append(box.height);
        result.append(detection);

    }

//    // 打印 JSON 输出
//    std::cout << result << std::endl;

    return input_image;
}


Json::Value post_process_json(Mat &&input_image, vector<Mat> &outputs, const vector<string> &class_name)
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += 85;
    }


    Json::Value result(Json::arrayValue);

    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);


        // create a Json object
        Json::Value detection(Json::objectValue);
        detection["label"] = class_name[class_ids[idx]];
        detection["confidence"] = confidences[idx];
        detection["box"] = Json::Value(Json::arrayValue);
        detection["box"].append(box.x);
        detection["box"].append(box.y);
        detection["box"].append(box.width);
        detection["box"].append(box.height);

        result.append(detection);

    }

    // 打印 JSON 输出
//    std::cout << result << std::endl;

    return result;
}

std::string extract_id(const std::string& image_path) {
    // Find the position of the last '.' (period) to get the file extension
    size_t dotPosition = image_path.rfind('.');

    if (dotPosition == std::string::npos) {
        // If no '.' was found, return an empty string
        return "";
    }

    // Find the position of the last '/' or '\\' to get the file name
    size_t slashPosition = std::max(image_path.rfind('/'), image_path.rfind('\\'));

    // Extract the file name, excluding the extension
    std::string fileName;
    if (slashPosition != std::string::npos) {
        fileName = image_path.substr(slashPosition + 1, dotPosition - slashPosition - 1);
    } else {
        // If no '/' or '\\' was found, the entire path is the file name
        fileName = image_path.substr(0, dotPosition);
    }

    // Remove leading zeros
    fileName.erase(0, std::min(fileName.find_first_not_of('0'), fileName.size() - 1));

    return fileName;
}





Json::Value infer_images(std::string image_paths, const vector<string> &class_name,  Net &net, int num_pics)
{

    int counter = 0;

    Json::Value image_detections(Json::arrayValue);

    for(const auto& file: fs::directory_iterator(image_paths))
    {
        if (counter >= num_pics)
        {
            break;
        }

        std::string image_path = file.path().string();

//        std::cout<<image_path<<std::endl;

        // use opencv to read the file
        Mat frame;
        frame = imread(image_path);


        vector<Mat> detections;
        detections = pre_process(frame, net);


        Json::Value image_detection = post_process_json(frame.clone(), detections, class_name);

//        image_detection["id"] = std::stod(extract_id(image_path));
//        std::cout<<"post processing Json: "<<image_detection<<std::endl;

        std::string string_id = extract_id(image_path);

//        std::cout<<"the id is: "<<string_id<<std::endl;

        Json::Value ID;

        ID["id"] = std::stod(string_id);

        // Put efficiency information.
        // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;

        Json::Value inference_time;

        inference_time["inference time"] = t;

        image_detection.append(inference_time);
        image_detection.append(ID);
        image_detections.append(image_detection);

        counter++;
    }

    return image_detections;
}



int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    std::string folder_path = argv[1];

//    std::string folder_path = "coco/images/val2017";

    std::string model_path = argv[2];

    std::string pic_nums_str = argv[3];

//    std::cout << "number of pics: " << pic_nums_str << std::endl;

    int pic_nums = std::stod(pic_nums_str);

    // Load class list.
    vector<string> class_list;
    ifstream ifs("coco.names");
    string line;

//    if (ifs.is_open()) {
//        // File was successfully opened.
//        // You can proceed with readin
//        cout<<"file is open!"<<endl;
//        }
//    else
//    {
//        cout<<"not open"<<endl;
//    }

    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }

    if (class_list.empty())
    {
        cerr << "Error: Class list is empty!" << endl;
        return -1;
    }

    // Load model.
    Net net;
    net = readNet(model_path);

    Json::Value images_inferences =infer_images(folder_path, class_list, net, pic_nums);

    // print JSON output
    std::cout << images_inferences << std::endl;

////    cout << "Loading image sample.jpg..." << endl;
//    // Load image.
//    Mat frame;
//    frame = imread(image_path);
////    frame = imread("D:\\Med_TA\\sample.jpg");
//    if (frame.empty())
//    {
//        cerr << "Error: Image not found!" << endl;
//        return -1;
//    }
//
//
//    // Load model.
//    Net net;
//    net = readNet(model_path);
////    net = readNet("models/yolov5s.onnx");
//
//    vector<Mat> detections;
//    detections = pre_process(frame, net);
//
//    Mat img = post_process(frame.clone(), detections, class_list);
////    Mat clonedFrame = frame.clone();
////    Mat img = post_process(clonedFrame, detections, class_list);
//
//    // Put efficiency information.
//    // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
//    vector<double> layersTimes;
//    double freq = getTickFrequency() / 1000;
//    double t = net.getPerfProfile(layersTimes) / freq;
//    string label = format("Inference time : %.2f ms", t);
//    putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
//
//
////    Json::Value inference(Json::objectValue);   // save the inference obj as json dict
//    result[0]["inference time"] = t;    // saved a double type num
//    result.append(inference);

//        // 打印 JSON 输出
//    std::cout << result << std::endl;

//    imshow("Output", img);
//    waitKey(0);
//
//    // Wait for a key press indefinitely
//    while (true) {
//        int key = cv::waitKey(0); // 0 means wait indefinitely
//
//        // Check if a specific key is pressed, e.g., 'q' or 'esc' to exit
//        if (key == 'q' || key == 27) { // 'q' or 27 (ASCII code for 'esc' key)
//            break; // Exit the loop and close the window
//        }
//    }

    return 0;
}