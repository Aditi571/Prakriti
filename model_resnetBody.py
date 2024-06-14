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
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load pre-trained ResNet model
resnet = models.resnet18(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in resnet.parameters():
    param.requires_grad = False

# Modify the last layer for 3 classes
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 3)

# Define transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = ImageFolder('/home/cvlab/Desktop/prakriti/Resnet/Label_PICS/BODY', transform=transform)

# Split dataset into train and test
train_data, test_data = torch.utils.data.random_split(dataset, [len(dataset) - len(dataset)//5, len(dataset)//5])

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    total_train = 0
    correct_train = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
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
    for images, labels in test_loader:
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
        # print(f'label predicted : {predicted}')
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('Correct prediciton :', correct)
print("Incorrect prediction : ", total - correct)
accuracy = correct / total
print('Accuracy:', accuracy)
