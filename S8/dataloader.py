import torch
import torchvision
import torchvision.transforms as transforms

def loadCiFAR10():
    train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomCrop(32),
     transforms.RandomRotation((-15.0, 15.0)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)


    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    SEED = 1
    # Check if CUDA is available
    cuda = torch.cuda.is_available()
    print("Is CUDA available: ", cuda)

    torch.manual_seed(SEED)

    if cuda:
      torch.cuda.manual_seed(SEED)

    # Data loader argumens for train and test
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # Train loader
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # Test loader
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
    
    return (trainloader, testloader)
    
