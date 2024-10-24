import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# Load the dataset
data = pd.read_csv('train_dataset.csv')

# Extract pixel values and labels
pixels = data['pixels'].tolist()
labels = data['emotion'].values

# Convert the pixel strings to 48x48 numpy arrays and normalize pixel values
def process_images(pixels):
    images = np.array([np.fromstring(pixel, sep=' ').reshape(48, 48) for pixel in pixels])
    images = images / 255.0  # Normalize the pixel values to be between 0 and 1
    images = images.reshape(images.shape[0], 48, 48, 1)  # Add channel dimension
    return images

images = process_images(pixels)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create a custom Dataset class
class EmotionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(np.uint8(self.images[idx].reshape(48, 48) * 255), mode='L')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Image augmentations and transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Data loaders
train_dataset = EmotionDataset(X_train, y_train, transform=transform)
val_dataset = EmotionDataset(X_val, y_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Simplified CNN model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()

        # CNN layers
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x

# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total * 100
        val_acc, val_loss = evaluate_model(model, val_loader, criterion)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%, Val Loss: {val_loss:.4f}")

# Evaluation function
def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculations during evaluation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total * 100
    avg_val_loss = val_loss / len(val_loader)
    return val_acc, avg_val_loss

# Training the model
epochs = 50
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=epochs)

# Load the test data
test_data = pd.read_csv('test.csv')
test_pixels = test_data['pixels'].tolist()
test_images = process_images(test_pixels)

# Create a DataLoader for test data
class TestDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(np.uint8(self.images[idx].reshape(48, 48) * 255), mode='L')
        if self.transform:
            image = self.transform(image)
        return image

test_dataset = TestDataset(test_images, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Generate predictions for the test data
def predict(model, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions

# Predict and save to submission.csv
predictions = predict(model, test_loader)
submission = pd.DataFrame({'id': test_data['id'], 'emotion': predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file created!")

