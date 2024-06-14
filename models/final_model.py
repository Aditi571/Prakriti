import os
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.metrics import accuracy_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  


excel_path = '../Data.xlsx'


df = pd.read_excel(excel_path)
dataset_labels = []

for index, row in df.iterrows():
    image_eye_path = '/home/cvlab/Desktop/prakriti/face_masking-main/res/test_res/' + row['AppID']
    if os.path.exists(image_eye_path):
        dataset_labels.append({
            'label': row['Prakriti'],
       })

print("Dataset length:", len(dataset_labels))

Y_labels = np.array([sample['label'] for sample in dataset_labels])
Y = np.zeros(len(Y_labels), dtype=int)
for i, label in enumerate(Y_labels):
    if label.startswith('P'):
        Y[i] = 0
    elif label.startswith('V'):
        Y[i] = 1
    elif label.startswith('K'):
        Y[i] = 2



class prediction:
    def __init__(self, model_path, excel_path,target_length,dim_val):
        self.loaded_model = tf.keras.models.load_model(model_path)
        self.excel_path = excel_path
        self.target_length = target_length
        self.dim_val=dim_val

    def load_data(self):
        df = pd.read_excel(self.excel_path)
        dataset = []

        for index, row in df.iterrows():
            image_eye_path = '/home/cvlab/Desktop/prakriti/face_masking-main/res/test_res/' + row['AppID'] +'/eye_colors.txt'
            
            if os.path.exists(image_eye_path): 
                eye_colors_data = []
                with open(image_eye_path, 'r') as file:
                    for line in file:
                        rgb_values = line.strip().split()
                        try:
                            rgb_values = [int(value) for value in rgb_values]
                            eye_colors_data.append(rgb_values)
                        except ValueError:
                            print(f"Ignoring non-integer RGB values in file: {image_eye_path}, line: {line}")
                    
                    padded_rgb_values = np.pad(eye_colors_data, ((0, self.target_length - len(eye_colors_data)), (0, 0)), mode='constant')
                    
                    dataset.append({
                        'RGB': padded_rgb_values,
                        'label': row['Prakriti'],
                        'AppId': row['AppID']
                    })

        self.dataset = dataset

    def preprocess_data(self):
        X = np.array([sample['RGB'] for sample in self.dataset])
        Id= np.array([sample['AppId'] for sample in self.dataset])
        Y_labels = np.array([sample['label'] for sample in self.dataset])

        Y = np.zeros(len(Y_labels), dtype=int)
        for i, label in enumerate(Y_labels):
            if label.startswith('P'):
                Y[i] = 0
            elif label.startswith('V'):
                Y[i] = 1
            elif label.startswith('K'):
                Y[i] = 2

        self.X = X
        self.Id = Id
        self.Y = Y

    def predict_with_cnn_model(self, X_data,dim_val):
        X_data_reshaped = X_data.reshape(X_data.shape[0], dim_val, dim_val, 1)
        predictions = self.loaded_model.predict(X_data_reshaped)
        predicted_labels = np.argmax(predictions, axis=1)
        return predicted_labels

    def predict(self,i):
        self.load_data()
        self.preprocess_data()
        X_sample_reshaped = self.X[i].reshape(1, self.dim_val, self.dim_val, 1)
        predicted_label = self.predict_with_cnn_model(X_sample_reshaped,self.dim_val)
        # print("Id:", self.Id[i])
        # print("Predicted Label by eye model ", i, ":", predicted_label)
        return predicted_label

            

# Usage
model_path_eye = 'eye_model.h5'
model_path_hair = 'hair_model.h5'
model_path_lip = 'lip_model.h5'
model_path_skin = 'skin_model.h5'


from collections import Counter

def Final_prediction(i):
    sum_0=0
    sum_1=0
    sum_2=0

    eye_predictor = prediction(model_path_eye, excel_path,2187,81)
    eye_label=eye_predictor.predict(i)
    if eye_label==0:
        sum_0+= 42.11
    if eye_label==1:
        sum_1+= 42.11
    if eye_label==2:
        sum_2+= 42.11

    hair_predictor = prediction(model_path_hair, excel_path,177147,729)
    hair_label=hair_predictor.predict(i)
    if hair_label==0:
        sum_0+= 57.89
    if hair_label==1:
        sum_1+= 57.89
    if hair_label==2:
        sum_2+= 57.89

    lip_predictor = prediction(model_path_lip, excel_path,19683 ,243)
    lip_label=lip_predictor.predict(i)
    if lip_label==0:
        sum_0+= 31.58
    if lip_label==1:
        sum_1+= 31.58
    if lip_label==2:
        sum_2+= 31.58

    skin_predictor = prediction(model_path_skin, excel_path,177147,729)
    skin_label=skin_predictor.predict(i)
    if skin_label==0:
        sum_0+= 42.11
    if skin_label==1:
        sum_1+= 42.11
    if skin_label==2:
        sum_2+= 42.11

    # final_array = [tuple(eye_label), tuple(hair_label), tuple(lip_label), tuple(skin_label)]
    # label_counts = Counter(final_array)

    # most_common_label = label_counts.most_common(1)[0][0]
    if max(sum_0,sum_1,sum_2)==sum_0:
        return 0
    elif max(sum_0,sum_1,sum_2)==sum_1:
        return 1
    elif max(sum_0,sum_1,sum_2)==sum_2:
        return 2

final_predicted_labels = []
for i in range(len(Y)):
    final_predicted_label = Final_prediction(i)

    final_predicted_labels.append(final_predicted_label)

print("lenght of final predited labels", len(final_predicted_labels))

accuracy = accuracy_score(Y, final_predicted_labels)
print("Overall Accuracy Score:", accuracy)