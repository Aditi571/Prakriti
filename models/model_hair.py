import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Step 1: Read Excel Sheet
excel_file = "../Data.xlsx"
df = pd.read_excel(excel_file)

# Step 2: Load Images and Labels
dataset = []
target_length = 177147  # Desired length of RGB arrays

for index, row in df.iterrows():
    image_eye_path = '/home/cvlab/Desktop/prakriti/face_masking-main/res/test_res/' + row['AppID'] +'/hair_colors.txt'
    
    if os.path.exists(image_eye_path): 
        eye_colors_data = []  # Initialize list for RGB values
        
        with open(image_eye_path, 'r') as file:
            for line in file:
                # Split the line into RGB values
                rgb_values = line.strip().split()
                try:
                    # Convert RGB values to integers
                    rgb_values = [int(value) for value in rgb_values]
                    eye_colors_data.append(rgb_values)
                except ValueError:
                    # Skip the line if it contains non-integer values
                    print(f"Ignoring non-integer RGB values in file: {image_eye_path}, line: {line}")
                
            # Pad or truncate RGB arrays to match the target length
            padded_rgb_values = np.pad(eye_colors_data, ((0, target_length - len(eye_colors_data)), (0, 0)), mode='constant')
            
            # Append RGB values to dataset
            dataset.append({
                'RGB': padded_rgb_values,
                'label': row['Prakriti'],
                'AppId': row['AppID']  # Replace 'Prakriti' with the actual name of your label column
            })
    else:
        print("File not found:", image_eye_path)

# Print dataset length for verification
print("Dataset length:", len(dataset))

# Convert the dataset into NumPy arrays
X = np.array([sample['RGB'] for sample in dataset])
Id = np.array([sample['AppId'] for sample in dataset])
Y_labels = np.array([sample['label'] for sample in dataset])

# Map labels to numerical values
Y = np.zeros(len(Y_labels), dtype=int)
for i, label in enumerate(Y_labels):
    if label.startswith('P'):
        Y[i] = 0
    elif label.startswith('V'):
        Y[i] = 1
    elif label.startswith('K'):
        Y[i] = 2

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Preprocess the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).astype(np.float32)
X_val_scaled = scaler.transform(X_val.reshape(X_val.shape[0], -1)).astype(np.float32)

# Reshape the data for CNN model
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 729, 729, 1)
X_val_scaled = X_val_scaled.reshape(X_val_scaled.shape[0], 729, 729, 1)

# Define CNN model
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(729, 729, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # Output layer with 3 neurons for 3 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the CNN model
cnn_model = create_cnn_model()
cnn_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_val_scaled, y_val))

# Evaluate the model
val_loss, val_accuracy = cnn_model.evaluate(X_val_scaled, y_val)
print("Validation Accuracy:", val_accuracy)

# Define function to predict with CNN model
def predict_with_cnn_model(model, X_data):
    # Reshape input data for the CNN model
    X_data_reshaped = X_data.reshape(X_data.shape[0], 729, 729, 1)
    # Perform predictions
    predictions = model.predict(X_data_reshaped)
    # Convert predictions to class labels
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels

# Save the model
cnn_model.save('hair_model.h5')

# Predict labels for the samples
for i in range(10):
    # Reshape the sample to match the input shape expected by the model
    X_sample_reshaped = X[i].reshape(1, 729, 729, 1)
    # Perform prediction using the CNN model
    predicted_label = predict_with_cnn_model(cnn_model, X_sample_reshaped)
    print("Id" , Id[i])
    print("Predicted Label for sample", i, ":", predicted_label)
