import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from kymatio.torch import Scattering2D
import kymatio.datasets as scattering_datasets

torch.cuda.empty_cache()

class Scattering2dVIT(nn.Module):
    '''
        ViT with scattering transform as input
    '''
    def __init__(self, scattering_output_channels, num_classes=10):
        super(Scattering2dVIT, self).__init__()
        self.scattering_output_channels = scattering_output_channels
        self.num_classes = num_classes
        self.build()

    def build(self):
        # Load a pre-trained ResNet model
        self.vit = models.vit_b_32(pretrained=True)
        
        # Modify the first convolution layer to accept the number of input channels from scattering transform
        self.vit.conv_proj = nn.Conv2d(self.scattering_output_channels, 768, kernel_size=32, stride=32)
        self.vit.heads = nn.Sequential(nn.Linear(768, 10, bias=True))
        
        for param in self.vit.parameters():
            param.requires_grad = False
            
        for param in self.vit.encoder.layers[-1:].parameters():
            param.requires_grad = True
            
        for param in self.vit.heads.parameters():
            param.requires_grad = True
                    
    def forward(self, x):
        # Flatten the extra dimension
        batch_size, channels, depth, height, width = x.size()
        x = x.view(batch_size, channels * depth, height, width)
        return self.vit(x)

def train(model, device, train_loader, optimizer, epoch, scattering):
    train_loss_log = np.zeros((1,16))
    train_time = np.zeros((1,1))
    
    train_start_time = time.time()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        scattering_output = scattering(data)
        optimizer.zero_grad()
        output = model(scattering_output)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            train_loss_log[0][batch_idx/50] = loss.item()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    train_end_time = time.time()
    train_duration = datetime.timedelta(seconds=train_end_time - train_start_time)

    train_time[0][0] = str(train_duration)
    
    print(f"Training time for epoch:{epoch} is {str(train_duration)}")

def test(model, device, test_loader, scattering):
    test_loss_log = np.zeros((1,1))
    test_accuracy = np.zeros((1,1))
    test_time = np.zeros((1,1))
    
    test_start_time = time.time()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(scattering(data))
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    
    test_end_time = time.time()
    test_duration = datetime.timedelta(seconds=test_end_time - test_start_time)

    test_loss_log[0][0] = test_loss
    test_accuracy[0][0] = 100. * correct / len(test_loader.dataset)
    test_time[0][0] = str(test_duration)
    
    print(f"Testing time for epoch:{epoch+1} is {str(test_duration)}")
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

mode = 2
image_size = 896

if mode == 1:
    scattering = Scattering2D(J=2, shape=(image_size, image_size), max_order=1)
    K = 17*3
elif mode == 2:
    scattering = Scattering2D(J=2, shape=(image_size, image_size))
    K = 81*3
else:
    scattering = Scattering2D(J=2, shape=(image_size, image_size))
    K = 81*3

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

scattering = scattering.to(device)
model = Scattering2dVIT(K).to(device)

# Wrap the model with DataParallel
if use_cuda and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# DataLoaders
num_workers = 4
pin_memory = True if use_cuda else False

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
datasets.CIFAR10(root='.', train=True, transform=transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomCrop(image_size, padding=4),
    transforms.RandomRotation(10),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    normalize,
]), download=True),
batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='.', train=False, transform=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

total_start_time = time.time()

num_epoch = 100

for epoch in range(0, num_epoch):
    train(model, device, train_loader, optimizer, epoch+1, scattering)
    test(model, device, test_loader, scattering)

    # write the log of training to the file
    with open('vit_b_32_scat.txt', 'W') as file:
        
    
    # Save the model every 20 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'ViT_Scat_epoch_{epoch+1}.pth')
        print(f'Model saved at epoch {epoch+1}')

total_end_time = time.time()
total_elapsed_time = datetime.timedelta(seconds=total_end_time - total_start_time)

print(f"\nTotal training time for {num_epoch} epochs: {str(total_elapsed_time)}")
