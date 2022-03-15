import torch
import numpy as np
from data import ImageWoofData,train_ds,val_ds
from torch.utils.data.sampler import SubsetRandomSampler



TEST_SIZE = 0.25
BATCH_SIZE = 64
SEED = 42


val_loader = torch.utils.data.DataLoader(val_ds,collate_fn=val_ds.collate_fn, batch_size=BATCH_SIZE)


num_train = len(train_ds)
indices = list(range(num_train))
split = int(np.floor(TEST_SIZE * num_train))
np.random.shuffle(indices)



train_idx, test_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

#75% data for supervised learning
trainloader = torch.utils.data.DataLoader(train_ds,sampler=train_sampler,collate_fn=train_ds.collate_fn, batch_size=32)
#25% data for pseudo labelling
testloader = torch.utils.data.DataLoader(train_ds,sampler=test_sampler,collate_fn=train_ds.collate_fn, batch_size=32)