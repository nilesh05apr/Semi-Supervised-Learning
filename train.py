from torchvision.transforms.transforms import ToPILImage
from torch_snippets import *
from model import Model
from utils import transform1,transform2,transform3,transform4,transform5
from utils import train_batch,validate_batch
from data import trn_dl,val_dl
from dataloader import trainloader,testloader,val_loader


model, loss_fn, optmizier = Model()
n_epochs = 5
log = Report(n_epochs)

for epoch in range(n_epochs):
    
    # supervised learning
    N = len(trainloader)
    for ix, data in enumerate(trainloader):
        train_loss,train_acc = train_batch(data, model, optmizier, loss_fn)
        log.record(epoch+(ix+1)/N, trn_loss=train_loss, end='\r')
        log.record(epoch+(ix+1)/N, trn_acc=train_acc, end='\r')

    #unsupervised learning
    N = len(testloader)
    for ix,data in enumerate(testloader):
        
      model.train()
      ims, targets = data
      ims, targets = ims.cuda(), targets.cuda()
      y_pred = model(ims)
      X_unl = []
      Y_unl = []

      for img in ims:
       t1 = transform1(img)
       t2 = transform2(img)
       t3 = transform3(img)
       t4 = transform4(img)
       t5 = transform5(img)
       X_unl.append(t1)
       X_unl.append(t2)
       X_unl.append(t3)
       X_unl.append(t4)
       X_unl.append(t5)
      for lbl in targets:
        for _ in range(5):
          Y_unl.append(lbl)
      X_unl = torch.stack(X_unl)
      Y_unl = torch.stack(Y_unl) 
      optmizier.zero_grad()
      predictions = model(X_unl.cuda())
      batch_loss = loss_fn(predictions, Y_unl)
      batch_loss.backward()
      optmizier.step()
      val_loss = batch_loss.item() 

    for ix, data in enumerate(val_loader):
        val_loss, accuracy = validate_batch(model, data, loss_fn)
        log.record(epoch+(ix+1)/N, val_acc=accuracy, end='\r')

    log.report_avgs(epoch+1)

log.plot_epochs(['val_acc', 'trn_acc'])
