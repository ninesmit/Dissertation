import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import pandas as pd
import csv
import time
import datetime
import argparse
import sys
import math

from kymatio.torch import Scattering2D
from torchvision import datasets
import kymatio.datasets as scattering_datasets

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

## Additional Function

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        init.ones_(m.weight)
        init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        init.xavier_uniform_(m.weight)

def get_some_weights(model, num_weights=10):
    weights = []
    for param in model.parameters():
        if param.requires_grad:
            weights.extend(param.view(-1).detach().cpu().numpy())
        if len(weights) >= num_weights:
            break
    return weights[:num_weights]

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
## Pre-norm Class
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
## Attention Class

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
## Transformer Class

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

## Scattering with ViT class
    
class Scattering2dVIT(nn.Module):
    '''
        ViT with scattering transform as input
    '''
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool, channels, dim_head, dropout, emb_dropout):
        super(Scattering2dVIT, self).__init__()
        self.image_size = int((image_size/4) ** 2)
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp = mlp_dim
        self.pool = pool
        self.channels = channels
        self.dim_head = dim_head
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.build()

    def build(self):
        
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = self.channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, self.dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout2 = nn.Dropout(self.emb_dropout)
        self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp, self.dropout)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )

    def forward(self, img):
        batch_size, channels, depth, height, width = img.size()
        img = img.view(batch_size, channels * depth, height, width)
        
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout2(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        
        return self.mlp_head(x)
    
## Train and Test function

def train(model, device, train_loader, optimizer, epoch):
    train_loss_log = np.zeros((1,16))
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
            train_loss_log[0][int(batch_idx/50)] = loss.item()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    train_end_time = time.time()
    train_duration = datetime.timedelta(seconds=train_end_time - train_start_time)
    train_time = str(train_duration)
    print(f"Training time for epoch:{epoch} is {train_time}")
    
    # Get and log some weights
    some_weights = get_some_weights(model)
    print(f"Epoch {epoch} weights: {some_weights}")
    
    return train_loss_log, train_time

def test(model, device, test_loader):
    
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

    test_loss_log = test_loss
    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_time = str(test_duration)
    
    print(f"Testing time for epoch:{epoch+1} is {test_time}")
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_loss_log, test_time, test_accuracy

## All hyperparamters

mode = 2
image_size = 128
patch_size = 4
text_file_name = 'Vit_Img_Model3.txt'
num_classes = 10
dim = 1024
depth = 6
heads = 8
mlp_dim = 1024
pool='cls'
# channels=1
dim_head=128
dropout = 0.05
emb_dropout = 0.05
num_workers = 4
batch_size = 128
learning_rate = 0.0001
num_epoch = 100

## Training Loop

if mode == 1:
    scattering = Scattering2D(J=2, shape=(image_size, image_size), max_order=1)
    K = 17*3
elif mode == 2:
    scattering = Scattering2D(J=2, shape=(image_size, image_size))
    K = 81*3
else:
    print("Specify the number of scale for scattering transformation")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

scattering = scattering.to(device)
model = Scattering2dVIT(image_size=image_size, 
                        patch_size=patch_size, 
                        num_classes=num_classes, 
                        dim=dim, 
                        depth=depth, 
                        heads=heads, 
                        mlp_dim=mlp_dim,
                        pool=pool,
                        channels=K,
                        dim_head=dim_head,
                        dropout=dropout, 
                        emb_dropout=emb_dropout).to(device)
model.apply(initialize_weights)

total_params = count_trainable_parameters(model)
print(f"Total trainable parameters: {total_params}")

with open(text_file_name, 'a') as file:
    file.write(f"""Number of parameter:{total_params}""")

if use_cuda and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# DataLoaders
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
batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='.', train=False, transform=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
def lr_lambda(epoch):
    if epoch < 70:
        return 1
    else:
        return 0.5

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

total_start_time = time.time()

acc_log = np.zeros((1,num_epoch))

for epoch in range(0, num_epoch):
    train_loss_log, train_time = train(model, device, train_loader, optimizer, epoch+1)
    test_loss_log, test_time, test_accuracy = test(model, device, test_loader)
    acc_log[0][epoch] = test_accuracy
  
    # Step the learning rate scheduler
    scheduler.step()
    
    # write the log of training to the file
    with open(text_file_name, 'a') as file:
        file.write(f"""\nEpoch: {epoch+1}\nTraining loss: {[train_loss_log[0][i] for i in range(15)]}\nTraining time: {train_time}\nAverage test loss:{test_loss_log}\nTest time: {test_time}\nAccuracy: {test_accuracy}\n""")

    # Save the model every 20 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'Vit_Img_Model3_epoch_{epoch+1}.pth')
        print(f'Model saved at epoch {epoch+1}')

total_end_time = time.time()
total_elapsed_time = datetime.timedelta(seconds=total_end_time - total_start_time)

with open(text_file_name, 'a') as file:
    file.write(f"""Total Training time:{str(total_elapsed_time)}""")

print(f"Highest Accuracy is {acc_log.max()} at epoch {np.argmax(acc_log) + 1}")
print(f"\nTotal training time for {num_epoch} epochs: {str(total_elapsed_time)}")
