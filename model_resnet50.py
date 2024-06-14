import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import pandas as pd
from torch.utils.data import random_split
# Load pre-trained ResNet model
resnet = models.resnet18(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in resnet.parameters():
    param.requires_grad = False

# Modify the first convolutional layer to accept 1 channel instead of 3
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


# Modify the last layer for 3 classes
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 3)
# Create ResNet50 model
resnet50 = models.resnet50(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in resnet50.parameters():
    param.requires_grad = False

# Modify the last layer for 3 classes
num_ftrs = resnet50.fc.in_features


# Define transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define paths to the folders containing face, hand, and body images
face_folder = '/home/cvlab/Desktop/prakriti/Resnet/Label_PICS/FACE'
hand_folder = '/home/cvlab/Desktop/prakriti/Resnet/Label_PICS/HAND'
body_folder = '/home/cvlab/Desktop/prakriti/Resnet/Label_PICS/BODY'

# Define transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets for each body part
face_dataset = ImageFolder(face_folder, transform=transform)
hand_dataset = ImageFolder(hand_folder, transform=transform)
body_dataset = ImageFolder(body_folder, transform=transform)

# Split datasets into training and testing sets
total_size = len(face_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size
# Perform random split on face dataset
face_train_dataset, face_test_dataset = random_split(face_dataset, [train_size, test_size])

# Define the sizes for the hand and body datasets based on the split of the face dataset
hand_train_dataset = torch.utils.data.Subset(hand_dataset, face_train_dataset.indices)
hand_test_dataset = torch.utils.data.Subset(hand_dataset, face_test_dataset.indices)

body_train_dataset = torch.utils.data.Subset(body_dataset, face_train_dataset.indices)
body_test_dataset = torch.utils.data.Subset(body_dataset, face_test_dataset.indices)


# Create data loaders
face_train_loader = DataLoader(face_train_dataset, batch_size=32, shuffle=False)
face_test_loader = DataLoader(face_test_dataset, batch_size=32)

hand_train_loader = DataLoader(hand_train_dataset, batch_size=32, shuffle=False)
hand_test_loader = DataLoader(hand_test_dataset, batch_size=32)

body_train_loader = DataLoader(body_train_dataset, batch_size=32, shuffle=False)
body_test_loader = DataLoader(body_test_dataset, batch_size=32)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    total_train = 0
    correct_train = 0
    for face_images, hand_images, body_images in zip(face_train_loader, hand_train_loader, body_train_loader):
        optimizer.zero_grad()
        # print(face_images[0].shape)
        # Get features from face images
        face_features = resnet50(face_images[0])
        # print(face_features.shape)
        # Get features from hand images
        hand_features = resnet50(hand_images[0])
        
        # Get features from body images
        body_features = resnet50(body_images[0])

        # Concatenate the features along the channel dimension
        concatenated_features = torch.cat((face_features, hand_features, body_features), dim=1)
        # Unsqueeze the concatenated features to make it a 4D tensor
        concatenated_features = concatenated_features.unsqueeze(1)
        concatenated_features = concatenated_features.unsqueeze(2)
        # Compute the outputs
        outputs = resnet(concatenated_features)
        # Obtain labels for the concatenated features
        labels = face_images[1]  # Assuming labels are the same for all datasets

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        # Calculate training accuracy
        _, predicted_train = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    # Calculate training accuracy for the epoch
    accuracy_train = correct_train / total_train
    print('Epoch:', epoch, 'Accuracy:', accuracy_train)

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for face_images, hand_images, body_images in zip(face_test_loader, hand_test_loader, body_test_loader):
        face_features = resnet50(face_images[0])
        # print(face_features.shape)
        # Get features from hand images
        hand_features = resnet50(hand_images[0])
        
        # Get features from body images
        body_features = resnet50(body_images[0])

        # Concatenate the features along the channel dimension
        concatenated_features = torch.cat((face_features, hand_features, body_features), dim=1)
        # Unsqueeze the concatenated features to make it a 4D tensor
        concatenated_features = concatenated_features.unsqueeze(1)
        concatenated_features = concatenated_features.unsqueeze(2)
        # Compute the outputs
        outputs = resnet(concatenated_features)
        # Obtain labels for the concatenated features
        labels = face_images[1]  # Assuming labels are the same for all datasets
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print('Accuracy:', accuracy)

