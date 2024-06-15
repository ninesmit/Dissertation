import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms, models
from kymatio.torch import Scattering2D
import kymatio.datasets as scattering_datasets

class Scattering2dDenseNet(nn.Module):
    '''
        DenseNet with scattering transform as input
    '''
    def __init__(self, in_channels, classifier_type='cnn'):
        super(Scattering2dDenseNet, self).__init__()
        self.in_channels = in_channels
        self.classifier_type = classifier_type
        self.build()

    def build(self):
        # Load a pre-trained ResNet model
        self.dense = models.densenet121(pretrained=True)
        
        # Modify the first convolution layer to accept the number of input channels from scattering transform
        self.dense.features = nn.Sequential(*list(self.dense.features[:4]) + list(self.dense.features[8:]))
        self.dense.features[0] = nn.Conv2d(self.in_channels, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.dense.features[1] = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
        # Replace the fully connected layer
        num_features = self.dense.classifier.in_features
        self.dense.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 10)
        )
                    

    def forward(self, x):
        # Flatten the extra dimension
        batch_size, channels, depth, height, width = x.size()
        x = x.view(batch_size, channels * depth, height, width)
        return self.dense(x)


def train(model, device, train_loader, optimizer, epoch, scattering):
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, scattering):
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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':

    """Train a simple Hybrid Scattering + ResNet model on CIFAR.

        Three models are demoed:
        'linear' - scattering + linear model
        'mlp' - scattering + MLP
        'cnn' - scattering + CNN

        scattering 1st order can also be set by the mode
        Scattering features are normalized by batch normalization.
        The model achieves around 88% testing accuracy after 10 epochs.

        scatter 1st order + linear achieves 64% in 90 epochs
        scatter 2nd order + linear achieves 70.5% in 90 epochs

        scatter + cnn achieves 88% in 15 epochs

    """
    image_size = 128
    mode = 2
    classifier = 'cnn'
    assert(classifier in ['linear', 'mlp', 'cnn'])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if mode == 1:
        scattering = Scattering2D(J=2, shape=(image_size, image_size), max_order=1)
        K = 17*3
    elif mode == 2:
        scattering = Scattering2D(J=2, shape=(image_size, image_size))
        K = 81*3
    else:
        scattering = Scattering2D(J=2, shape=(image_size, image_size))
        K = 81*3
        
    scattering = scattering.to(device)

    model = Scattering2dDenseNet(K, classifier).to(device)

    # DataLoaders
    num_workers = 4
    pin_memory = True if use_cuda else False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=scattering_datasets.get_dataset_dir('CIFAR'), train=True, transform=transforms.Compose([
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
        batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=scattering_datasets.get_dataset_dir('CIFAR'), train=False, transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(0, 150):
        train(model, device, train_loader, optimizer, epoch+1, scattering)
        test(model, device, test_loader, scattering)
