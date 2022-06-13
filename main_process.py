from glob import glob
import cv2
from utils import *
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import os
import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import time
from  torchvision import models
import torch.nn as nn
import torch
from torchvision import transforms


class Main_Process:
     
    def __init__(self, images_folder):
        self.images_folder = images_folder
        self.model = None
        self.input_shape = (224, 224, 3)
        self.n_classes = 2
        with open(os.path.join('weights', 'classes_ids.json'), 'r') as file:
            self.classnames_ids = json.load(file)
    
    def __loadNetworkModel(self):
        """ self.mobileNetV2 = tf.keras.applications.mobilenet_v2.MobileNetV2(
                            input_shape=self.input_shape,
                            alpha=1.0,
                            include_top=False,
                            weights='imagenet',
                            pooling='avg')
        

        #add a classifier to mobilenetv2
        self.model = Sequential()
        self.model.add(self.mobileNetV2)
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Dense(self.n_classes, activation='softmax')) """
        self.model = models.mobilenet_v2()
        #self.mobileNetV2.eval() 
        # reshape the last layer since imageNet contains 1000 class and we are doing a binary classification
        #num_ftrs = self.mobileNetV2.classifier[1].in_features
       # self.mobileNetV2.classifier[1] = nn.Linear(num_ftrs, 2)
        
       
         #load model weights
        #self.model.load_weights(os.path.join('weights', 'nn', 'nn_model_pytorch'))
        self.model=torch.load(os.path.join('weights', 'nn', 'nn_model_pytorch.pth'))

    def image_binarization(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert img.ndim == 2

        bin_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23,9)
        _, bin_img = cv2.threshold(bin_img, 127, 255,1)
        return bin_img
    
    def contours_detection(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
        
        #remove countours with one coordinate.
        filtered_contours =[]
        for cnt in contours:
            if len(cnt) > 1 :
                filtered_contours.append(cnt)
        
        return filtered_contours
    
    def regroup_contours_with_DBScan(self, contours):
        #From list of points to array of points
        contours_list = list(contours)
        list_points = [l.tolist() for l in contours_list]
        flat_list = [item for sublist in list_points for item in sublist]
        X = np.array(flat_list)

        #reshape the data points
        nsamples, nx, ny = X.shape
        train_dataset = X.reshape((nsamples,nx*ny))

        #clustering
        dbscan = DBSCAN(eps = 65, min_samples = 20)
        model = dbscan.fit(train_dataset)
        labels = model.labels_

        clusters = []
        for cluster_ in range(max(labels)):
            clusters.append(X[labels==cluster_])
        return clusters



    def draw_rectangles(self, clusters, img, image_label , load_choice, start):
            #start = time.time()
            frame_list=[]
            min_max=[]
            preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
            for cluster in clusters:
                min_X,max_X,min_Y,max_Y= calculate_min_max(cluster)
                area = cv2.contourArea(cluster)
                
                if(area > 50):
                    frame = img[min_Y-10:max_Y+10, min_X-10:max_X+10]
                    #frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA) #resize the frame
                    #frame = frame/255
                    frame = preprocess(frame)
                    #frame=frame.reshape(1, 224, 224, 3)
                    frame_list.append(frame.reshape((1, 3, 224, 224)))
                    frame_lists=np.vstack(frame_list)
                    cv2.rectangle(img, (min_X, min_Y), (max_X, max_Y), (255,255,0), 3)
                    min_max.append([min_X,min_Y])
            if load_choice == 'nn':
                #y_pred = np.argmax(self.model.predict(frame_lists), axis=1)
                print('ok')
                with torch.no_grad():
                    frame_lists=torch.Tensor(np.array(frame_lists))#badaltha c bon , jet error okhra taaw
                    print('size : ', frame_lists.size())
                    y_pred = np.argmax(self.model.predict(frame_lists), axis=1)
                #print(y_pred)
            # predicted_anomaly = self.classnames_ids[str(y_pred[0])]
            elif load_choice=='composed':
                y_pred = self.composedModelInference(frame)
            # predicted_anomaly = self.classnames_ids[str(y_pred)]
            end = time.time()
            inf_time = end-start
            print('inference time execution :',inf_time)
            i=0
            for value in y_pred:
                if value==0 :
                    labela= "Flush"
                else : labela = "Not Flush"
                #predicted_anomaly = self.classnames_ids[str(value)]
                cv2.putText(img, labela, (min_max[i][0], min_max[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                i+=1
                #cv2.drawContours(img, [box], 0, (255,255,255), 2)*
            os.makedirs('results', exist_ok=True)
            cv2.imwrite('results/'+image_label+'.png', img)

    def exc_pipeline(self,load_choice):
        if load_choice == 'nn':
            self.__loadNetworkModel()
            #self.model.save(os.path.join('weights','neural'))
        elif load_choice == 'composed':
            self.__loadcomposedModelPipeline()
        for image_path in tqdm(glob(self.images_folder+'/*.bmp')):
            image_label = image_path.split('\\')[-1].split('.')[0]
            img = cv2.imread(image_path)
            start = time.time()
            croped = image_cropping(img)
            img_bin = self.image_binarization(croped)
            contours = self.contours_detection(img_bin)
            anomalies = self.regroup_contours_with_DBScan(contours)
            #end = time.time()
            #inf_time = end-start
            self.draw_rectangles(anomalies, croped, image_label,load_choice, start)
            
            
if __name__ == '__main__':
    images_folder = 'images'
    main_process = Main_Process(images_folder)
    main_process.exc_pipeline(load_choice='nn')
