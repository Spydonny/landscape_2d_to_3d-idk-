import os
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import deep_lab_v3_resnet50 as dplb
from torchvision.io import read_image


#### Примерная структура датасета #################################
class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, plt3d_matrix_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.plt3d_matrix_dir = plt3d_matrix_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        plt3d_matrix_path = os.path.join(self.plt3d_matrix_dir, self.img_labels.iloc[idx, 1])
        plt3d_matrix = pd.read_csv(plt3d_matrix_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            plt3d_matrix = self.target_transform(plt3d_matrix)
        return image, plt3d_matrix
############################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = dplb.custom_DeepLabv3(out_channel=1)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#TODO: data for dataset
train_dataset = CustomDataset()
val_dataset = CustomDataset()

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, plt3d_matrixs in train_loader:
        images = images.to(device)
        plt3d_matrixs = plt3d_matrixs.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, plt3d_matrixs)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    scheduler.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, plt3d_matrixs in val_loader:
            images = images.to(device)
            plt3d_matrixs = plt3d_matrixs.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, plt3d_matrixs)
            val_loss += loss.item() * images.size(0)

    val_loss = val_loss / len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}')

    # Save the model checkpoint
    torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')

print('Training complete.')