import torch
import torchvision
import torchvision.transforms as transforms

def loadCiFAR10(aug=None):
    train_transform = aug
    if train_transform is None:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]
        )
        
        
    test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]
    )

    trainset = torchvision.datasets.ImageFolder(root='./tiny-imagenet-200/',
                                        transform=train_transform)
    print("TrainSet", trainset)

    testset = torchvision.datasets.ImageFolder(root='./tiny-imagenet-200/val/',
                                       transform=test_transform)
    
    print("TestSet", testset)

    SEED = 1
    # Check if CUDA is available
    cuda = torch.cuda.is_available()
    print("Is CUDA available: ", cuda)

    torch.manual_seed(SEED)

    if cuda:
      torch.cuda.manual_seed(SEED)

    # Data loader argumens for train and test
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=32)

    # Train loader
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    print("Trainloader", trainloader)
    # Test loader
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
    
    return (trainloader, testloader)
    
