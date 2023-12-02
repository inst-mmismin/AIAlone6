import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor

def get_transform(args): 
    transform = Compose([Resize((args.image_size, 
                                args.image_size)), 
                        ToTensor()])
    return transform 

def get_datasets(transform): 
    train_val_dataset = MNIST(root='../../data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = random_split(train_val_dataset, 
                                            [50000, 10000], 
                                            torch.Generator().manual_seed(42))
    test_dataset = MNIST(root='../../data', train=False, download=True, transform=transform)
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(args): 
    # 데이터 불러오기 
    # dataset 만들기 & 전처리하는 코드도 같이 작성 
    transform = get_transform(args)
    train_dataset, val_dataset, test_dataset = get_datasets(transform)
    # dataloader 만들기 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader



def get_model(args): 
    from networks.mlps import myMLP
    model = myMLP(image_size=args.image_size, 
                hidden_size=args.hidden_size, 
                num_class=args.num_class).to(args.device) 
    return model 

def get_loss(): 
    return loss 

def get_optim():
    return optim 