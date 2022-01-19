"""
@author: Ronak
"""

# Import needed libraries
import numpy as np
from numpy.core.numeric import Inf 
import pandas as pd 
import cv2
import time
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
import os
import argparse

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession


# Load labels
labels = pd.read_csv('../Required_files/signnames.csv')
# print(labels.head())

# Load mean image use for preprocessing further (mean image substraction)
os.chdir('E:/Study_3/project_git/Traffic-Sign-Detection-and-Classification/Required_files')
with open('mean_image_gray.pickle', 'rb') as f:
    mean = pickle.load(f)


class Inference():
    def __init__(self, input, weights, model, output=None):
        self.cfg = '../Required_files/yolov3_ts_test.cfg'
        self.probability_minimum = 0.2
        self.threshold = 0.2
        self.labels = labels
        self.model = model
        self.colours = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')

        # Checking input
        if os.path.isfile(input):
            # Further functionality needs to be added for Folder Inference :))
            if input[-4:] in ['.png', '.jpg']:
               self.input = input
               self.inference_mode = 'SingleImage'
            elif input[-4:] in ['.mp4', '.mkv', '.avi']:
                self.input = input
                self.inference_mode = 'Video'
            else:
                print("Invalid input file. The file should be an image or a video !!")
                exit(-1)
        else:
            print("Input file doesn't exist. Check the input path")
            exit(-1)
        
        # Checking weights
        if os.path.isfile(weights):
            if weights[-8:] == '.weights':
                self.weights = weights
            else:
                print(f"Invalid Weights file. {weights} does not end with '.weights'")
                exit(-1)
        else:
            print("Model weights file does not exist. Check the weights path")
            exit(-1)
        
        # Checking output
        if output == None:
            self.output = self.input.split('/')[-1]
        else:
            self.output = output
        
        # Loading Classifier_Model
        self.model = load_model(self.model)

        # Loading Yolov3 trained model
        self.network = cv2.dnn.readNetFromDarknet(self.cfg, weights)

        # use with GPU
        # self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        # Getting detection layers from all layers
        self.layers_all = self.network.getLayerNames()
        self.layers_names_output = [self.layers_all[i - 1] for i in self.network.getUnconnectedOutLayers()]

        self.runInference()

    def Preprocessor(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        
        return img
    
    def runInference(self):
        image_BGR = cv2.imread(self.input)
        h, w = image_BGR.shape[:2]

        t = 0
        # Blob from current frame
        blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Forward pass with blob through output layers
        self.network.setInput(blob)
        start = time.time()
        output_from_network = self.network.forward(self.layers_names_output)
        end = time.time()

        # Time
        t += end - start
        # print('Total amount of time {:.5f} seconds'.format(t))

        # Lists for detected bounding boxes, confidences and class's number
        bounding_boxes = []
        confidences = []
        class_numbers = []

        # Going through all output layers after feed forward pass
        for result in output_from_network:
            # Going through all detections from current output layer
            for detected_objects in result:
                # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]
                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)
                # Getting value of probability for defined class
                confidence_current = scores[class_current]

                # Eliminating weak predictions by minimum probability
                if confidence_current > self.probability_minimum:
                    # Scaling bounding box coordinates to the initial frame size
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Getting top left corner coordinates
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)
        
        # Implementing non-maximum suppression of given bounding boxes
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, self.probability_minimum, self.threshold)
        
        # Checking if there is any detected object been left
        if len(results) > 0:
            # Going through indexes of results
            for i in results.flatten():
                # Bounding box coordinates, its width and height
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                    
                    
                # Cut fragment with Traffic Sign
                c_ts = image_BGR[y_min:y_min+int(box_height), x_min:x_min+int(box_width), :]
        
                    
                if c_ts.shape[:1] == (0,) or c_ts.shape[1:2] == (0,):
                    pass
                else:
                    c_ts = self.Preprocessor(c_ts)
                    # Getting preprocessed blob with Traffic Sign of needed shape
                    blob_ts = cv2.dnn.blobFromImage(c_ts, 1 / 255.0, size=(32, 32), swapRB=False, crop=False)
                    blob_ts = blob_ts.transpose(0, 2, 3, 1)

                    # Feeding to the Keras CNN model to get predicted label among 43 classes
                    scores = self.model.predict(blob_ts)

                    # Scores is given for image with 43 numbers of predictions for each class
                    # Getting only one class with maximum value
                    prediction = np.argmax(scores)

                    # Colour for current bounding box
                    colour_box_current = self.colours[class_numbers[i]].tolist()
                    
                    # Drawing bounding box on the original current frame
                    image_BGR = cv2.rectangle(image_BGR, (x_min, y_min),
                                             (x_min + box_width, y_min + box_height),
                                             colour_box_current, 2)

                    # Preparing text with label and confidence for current bounding box
                    text_box_current = '{}: {:.4f}'.format(labels['SignName'][prediction],
                                                        confidences[i])
                    (w1, h1), _ = cv2.getTextSize(text_box_current, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    
                    # Putting text with label and confidence on the original image
                    image_BGR = cv2.rectangle(image_BGR, (x_min, y_min - 20), (x_min + w1, y_min), colour_box_current, -1, cv2.LINE_AA)
                    image_BGR = cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

                    cv2.imwrite(self.output, image_BGR)

        print(f'\n Detection finished in {t} seconds')
    
    def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', type=str, default=None, help='Path to input file(s):(.jpg)/(.png)')
        parser.add_argument('--weights', type=str, default=None, help='path to the model weights')
        parser.add_argument('--model', type=str, default=None, help='path to the classifier model(s)')
        parser.add_argument('--output', type=str, default=None, help='pathe to save result(s)')
        opt = parser.parse_args()
        return opt

    def main(opt):
        Inference(**vars(opt))


if __name__ == "__main__":

    opt = Inference.parse_opt()
    Inference.main(opt)

    # Inference(
    #     r'E:\Study_3\project_git\img_1.jpg',
    #     r'E:\Study_3\project_git\yolov3_ts_train_last.weights',
    #     r'E:\Study_3\project_git\new_model.h5',
    #     r'E:\Study_3\project_git\result_1.jpg'
    # )
