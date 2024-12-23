
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from torch import optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        #images will be 52x52 size
        #After 1st convolution, 50x50, after max pool, 25x25, after 2nd convolution, 22x22, after max pool, 11x11
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 4)
        self.hidden1 = nn.Linear(10 * 11 * 11, 300)
        self.hidden2 = nn.Linear(300, 100)
        self.output = nn.Linear(100, 6)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.hidden1(out)
        out = self.relu(out)
        out = self.hidden2(out)
        out = self.relu(out)
        out = self.output(out)
        return out


    def get_dataloader(self, resolution=3024):
        transform = transforms.Compose([
            transforms.ToTensor(), #convert to a tensor
            #transforms.Lambda(torch.flatten) #flatten to vectors
            transforms.Resize((resolution, resolution))
        ])

        #about 5000 images in train dataset
        batch_size  = 1
        train_dataset = ImageFolder(root='../data/train', transform=transform)

        #come back and change shuffle to true
        train_loader  = DataLoader(train_dataset, batch_size=batch_size,num_workers=0, pin_memory=True)
        return train_loader

    def train_model(self, train_dataloader, label_counts):

        device = "cuda" if torch.cuda.is_available() else "cpu" #Set device to gpu or cpu


        model      = self
        lr         = 0.001
        num_epochs = 100
        train_loss = []
        train_err  = []

        #Tensor containing number of samples per class
        label_counts = torch.tensor(list(label_counts.values())) 
        print(label_counts)
        class_weights = 1.0 / label_counts.float()
        class_weights /= class_weights.sum()

        optimizer  = optim.Adam(model.parameters(), lr=lr,weight_decay=0)
        loss_func = nn.CrossEntropyLoss(weight=class_weights)

        # Start loop
        for epoch in range(num_epochs): #(loop for every epoch)
            model.train()    # training mode
            running_loss = 0.0   
            running_error = 0.0 

            for i, (X, Y) in enumerate(train_dataloader): # load a batch data of images

                #Move batch to device if needed
                X = X.to(device)
                Y = Y.to(device) 

                optimizer.zero_grad() #Zero the gradient
                P = model.forward(X)  #Compute predicted probabilities
                _, Yhat = torch.max(P, 1)          #Compute predictions

                loss = loss_func(P, Y)

                loss.backward()       #Compute the gradient of the loss
                optimizer.step()      #Take a step

                # update running loss and error
                running_loss  += loss.item() * X.shape[0]
                running_error += torch.sum(Yhat.flatten() != Y.flatten()).item()

            #Compute loss for the epoch
            epoch_loss  = running_loss /  len(train_dataloader.dataset)
            epoch_error = running_error / len(train_dataloader.dataset) 

            # Append result
            train_loss.append(epoch_loss)
            train_err.append(epoch_error)

            # Print progress
            print('[Train #{}] Loss: {:.8f} Err: {:.4f}'.format(epoch+1, epoch_loss, epoch_error))

    def test_on_train(self):
        transform = transforms.Compose([
            transforms.Resize((52, 52)),
            transforms.ToTensor(), #convert to a tensor
        ])
        #5484 is size of train dataset
        train_dataset = ImageFolder(root='../data/train', transform=transform)
        train_loader  = DataLoader(train_dataset, batch_size=5484, shuffle=True,num_workers=0, pin_memory=True)

        for i, (X, Y) in enumerate(train_loader): # load a batch data of images

            P = self.forward(X)  #Compute predicted probabilities
            _, Yhat = torch.max(P, 1)          #Compute predictions

            #print error rate
            print(f"final training error rate: {torch.sum(Yhat.flatten() != Y.flatten()).item() / len(train_loader.dataset)}")
    