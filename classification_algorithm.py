from calendar import EPOCH
from glob import glob
from re import A
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from  torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import Accuracy
from torchsample.modules import ModuleTrainer

from torchvision import transforms
from PIL import Image
from torchsample.callbacks import EarlyStopping



import tensorflow as tf

import pickle
import json
from tensorflow.python.compiler.tensorrt import trt_convert as trt


class ClassificationModel:
    
    def __init__(self, n_classes=2, n_components=20, input_shape=(224, 224, 3), epochs=100, batch_size=2, learning_rate=0.001, train_choice='nn', splitBy='image', test_size=0.2, stratify=True):
        self.n_components = None
        self.mobileNetV2 = None
        self.model = None
        self.svm = None
        self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.classnames_ids = {}
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.train_choice = train_choice #composed model / neuralNetwork
        self.splitBy = splitBy #random / image
        self.stratify = stratify
        self.test_size = test_size

    def __features_transformation(self, features):
        pca = PCA(n_components= self.n_components)
        extracted_features = self.mobileNetV2.predict(features)
        reduced_features = pca.fit_transform(extracted_features)
        return reduced_features 

    def __loadDataSet(self):
        resize_shape = self.input_shape[:2]
        preprocess = transforms.Compose([ transforms.ToPILImage() , transforms.Resize((224, 224)) ,
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for i, class_name in enumerate(glob(os.path.join('dataset','*'))):
            self.classnames_ids[i] = class_name.split('\\')[-1]  #register new id -> class name
            
            #load training data
            train_path = os.path.join(class_name, 'train')
            for frame_path in glob(os.path.join(train_path,'*.bmp')):
                frame = cv2.imread(frame_path)
                #frame = cv2.resize(frame, resize_shape, interpolation = cv2.INTER_AREA) #resize the frame
                #frame = frame/255,
                #frame = torch.Tensor(frame)
                frame = preprocess(frame)
                #print(frame.size())
                self.X_train.append(frame.reshape((1, 3, 224, 224)))
                self.y_train.append(torch.Tensor([i]))

            #load test data
            test_path = os.path.join(class_name, 'test')
            for frame_path in glob(os.path.join(test_path, '*.bmp')):
                frame = cv2.imread(frame_path)
                #frame = cv2.resize(frame, resize_shape, interpolation = cv2.INTER_AREA) #resize the frame
                #frame = frame/255
                #frame = torch.Tensor(frame)
                frame = preprocess(frame)
                #print(frame.shape)
                #break
                self.X_test.append(frame.reshape((1, 3, 224, 224)))
                self.y_test.append(torch.Tensor([i]))
            #break       
        #split frames randomly and not by cup images (split already done during the dataset generation)
        
        if self.splitBy == 'random':
            dataset = self.X_train + self.X_test
            targets = self.y_train + self.y_test
            if self.stratify:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset, targets, test_size=self.test_size, stratify=targets, random_state=2022)
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset, targets, test_size=self.test_size, random_state=2022)
        
        #convert train/test data into numpy arrays
        self.X_train = torch.cat(self.X_train, axis=0)
        self.X_test = torch.cat(self.X_test, axis=0)
        self.y_test = torch.cat(self.y_test, axis=0).long()
        self.y_train = torch.cat(self.y_train, axis=0).long()
        #self.y_test = torch.nn.functional.one_hot(self.y_test, 2)
        #self.y_train = torch.nn.functional.one_hot(self.y_train, 2)



        #print(self.X_train[10].size())
        print('data loaded')
        print(self.X_test.size())
        print(self.X_train.size())
        print(self.y_test.size())
        print(self.y_train.size())
        
        with open(os.path.join('weights','classes_ids.json'), 'w') as file:
            json.dump(self.classnames_ids, file)
        
    def __initNeuralNetworkModel(self):
        self.mobileNetV2 = models.mobilenet_v2(pretrained=True)
        self.mobileNetV2.eval() 
        # reshape the last layer since imageNet contains 1000 class and we are doing a binary classification
        num_ftrs = self.mobileNetV2.classifier[1].in_features
        self.mobileNetV2.classifier[1] = nn.Linear(num_ftrs, 2)
        
        #freez all layers except last 4 layers ;).
        self.mobileNetV2.trainable = True
        for layer in self.mobileNetV2.features[:-4].parameters():
            layer.requires_grad = False
        
        #add a classifier to mobilenetv2
        '''
        self.model = nn.Sequential()
        self.model.append(self.mobileNetV2)
        self.model.append(nn.Dropout(0.2))
        self.model.append(nn.ReLU(256))
        self.model.append(nn.Dropout(0.1))
        self.model.append(nn.Sigmoid())
        #self.model = self.mobileNetV2
        '''
        #print(self.model)

        #compile the model
        optimizer = optim.Adam(self.mobileNetV2.parameters(),lr=self.learning_rate)
        loss = nn.CrossEntropyLoss()
        #metrics = SparseCategoricalAccuracy()
        self.trainer = ModuleTrainer(self.mobileNetV2)
        self.trainer.compile(optimizer=optimizer, loss=loss)
    '''
    def __initComposedModel(self):
        self.mobileNetV2 = tf.keras.applications.mobilenet_v2.MobileNetV2(
                            input_shape=self.input_shape,
                            alpha=1.0,
                            include_top=False,
                            weights='imagenet',
                            pooling='avg')
        
        self.svm = SVC(C=1.0, kernel='poly', degree=4, gamma='scale', coef0=0.9, shrinking=True, probability=False, tol=0.001, cache_size=200,
                       class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=2022)
    ''' 
    def __execute_neuralNetworkPipeline(self):
        self.__initNeuralNetworkModel()
        self.__loadDataSet()
        #y_train = np.asarray(self.y_train, dtype=np.float64)
        print(len(self.y_train))
        
        self.trainer.fit(self.X_train, self.y_train, None, self.batch_size, num_epoch= 100)
        

        callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
        self.trainer.set_callbacks(callbacks)

        #test_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test)).batch(32)
        #self.y_pred = np.argmax(self.model.predict(test_dataset), axis=1)
        #save the model
        os.makedirs(os.path.join('weights', 'nn'), exist_ok=True)
        #self.trainer.save_weights(os.path.join('weights', 'nnn', 'nn_model'))
        #self.trainer.save(os.path.join('weights', 'nn','nn_model'))
        dummy_input = torch.randn(1, 3, 224, 224)
        input_names = [ "actual_input" ]
        output_names = [ "output" ]

        torch.save(self.trainer,os.path.join('weights', 'nn','nn_model_pytorch.pth'))
        torch.onnx.export(self.mobileNetV2,
                  dummy_input, 
                  "model.onnx",
                  input_names=input_names,
                  output_names=output_names,
                  verbose=False,
                  export_params=True,
                  )
        
        #self.model = torch.load(os.path.join('weights', 'nn', 'nn_model_pytorch.pth'))
        #print('Converting to TF-TRT FP32...')
        #converter = trt.TrtGraphConverterV2(input_saved_model_dir='weights/neural/')
        #converter.convert()
        # Save the converted model
        #converter.save('weights/nn_model/model.trt')
        #print('conversion done')
    
        # Load converted model and infer
        #model = tf.saved_model.load(r'C:\Users\zeine\Desktop\sasano_cup\sasano_cup\weights\nn_model')
    '''
    def __execute_composedModelPipeline(self):
        self.__initComposedModel()
        self.__loadDataSet()
        X_train = self.__features_transformation(self.X_train)
        X_test = self.__features_transformation(self.X_test)
        self.svm.fit(X_train, self.y_train)
        self.y_pred = self.svm.predict(X_test)
        #save model
        os.makedirs(os.path.join('weights', 'composed'), exist_ok=True)
        pickle.dump(self.svm, open(os.path.join('weights', 'composed', 'composed_model.sav'), 'wb'))
    '''   
    def show_classification_report(self):
        print(classification_report(self.y_test, self.y_pred))
            
    def show_results(self):
        for i, frame in enumerate(self.X_test):
            class_id_pred = self.classnames_ids[self.y_pred[i]]
            class_id_true = self.classnames_ids[int(self.y_test[i])]
            if class_id_true==self.classnames_ids[0]:
                plt.title('True: {} Predicted: {}'.format(class_id_true, class_id_pred))
                plt.imshow(frame)
                plt.show()
        
    def train(self):
        if self.train_choice == 'nn':
            self.__execute_neuralNetworkPipeline()
        elif self.train_choice == 'composed':
            self.__execute_composedModelPipeline()
           
            
if __name__ == '__main__':
    model = ClassificationModel(epochs=100, train_choice='nn', splitBy='random')
    model.train()
    #model.show_classification_report()
    #model.show_results()