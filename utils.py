import torch
import copy
from data import train_ds,val_ds
from data import trn_dl,val_dl
from torch_snippets import *
from model import Model

model, loss_fn, optimizer = Model()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_batch(data, model, optimizer, loss_fn):
    model.train()
    ims, targets = data
    optimizer.zero_grad()
    predictions = model(ims)
    batch_loss = loss_fn(predictions, targets)
    batch_loss.backward()
    optimizer.step()
    acc = (torch.max(predictions, 1)[1] == targets).float().mean()
    return batch_loss.item(),acc

@torch.no_grad()
def validate_batch(model, data, loss_fn):
    model.eval()
    ims, labels = data
    _preds = model(ims)
    loss = loss_fn(_preds, labels)
    acc = (torch.max(_preds, 1)[1] == labels).float().mean()
    return loss.item(), acc




transform1 = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(10),  
        transforms.RandomAffine(0,shear=10,scale=(0.8,1.2)),  
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),  
        transforms.ToTensor(),  
        ])

transform2 = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(20),  
        transforms.RandomAffine(0,shear=10,scale=(0.75,1.25)),  
        transforms.ColorJitter(brightness=0.2,contrast=0.4,saturation=0.1),  
        transforms.ToTensor(),  
        ])

transform3 = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(25),  
        transforms.RandomAffine(0,shear=8,scale=(0.6,1.0)),  
        transforms.ColorJitter(brightness=0.1,contrast=0.3,saturation=0.15),  
        transforms.ToTensor(),  
        ])

transform4 = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(35),  
        transforms.RandomAffine(0,shear=10,scale=(0.8,1.29)),  
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),  
        transforms.ToTensor(),  
        ])

transform5 = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(90),  
        transforms.RandomAffine(0,shear=10,scale=(0.95,1.7)),  
        transforms.ColorJitter(brightness=0.45,contrast=0.15,saturation=0.25),  
        transforms.ToTensor(),  
        ])
