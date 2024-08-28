import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import deep_lab_v3_resnet50 as dplb


#### Eto huinya #################################
class CustomDataset(Dataset):
    def __init__(self, images, plt3d, transform=None):
        self.images = images
        self.plt3d = plt3d
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.plt3d[idx]
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
############################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = dplb.custom_DeepLabv3(out_channel=1)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#TODO: data for dataset
train_dataset = CustomDataset(images=0, plt3d=0, transform=transforms.ToTensor())
val_dataset = CustomDataset(images=0, plt3d=0, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, plt3d in train_loader:
        images = images.to(device)
        plt3d = plt3d.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, plt3d)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    scheduler.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, plt3d in val_loader:
            images = images.to(device)
            plt3d = plt3d.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, plt3d)
            val_loss += loss.item() * images.size(0)

    val_loss = val_loss / len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}')

    # Save the model checkpoint
    torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')

print('Training complete.')