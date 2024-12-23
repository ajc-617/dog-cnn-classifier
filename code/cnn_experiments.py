import os
import pandas as pd
from cnn import CNN
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from torch import optim


bentley_path = "../data/train/bentley"
panda_path = "../data/train/panda"

print(len(os.listdir(bentley_path)))
print(len(os.listdir(panda_path)))

exit(0)
myCNN = CNN()
first_dataloader = myCNN.get_dataloader()
X1,Y1 = next(iter(first_dataloader))


second_dataloader = myCNN.get_dataloader(resolution=500)
X2,Y2 = next(iter(second_dataloader))


# Create a figure
plt.figure(figsize=(10, 5))  # Adjust figure size to fit the images

# Plot the first image
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.imshow(X1[0].permute(1,2,0).numpy())
plt.title("Image 1")
plt.axis("off")  # Turn off axis

# Plot the second image
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.imshow(X2[0].permute(1,2,0).numpy())
plt.title("Image 2")
plt.axis("off")  # Turn off axis

# Display the plot
plt.tight_layout()
plt.show()

exit(0)
myCNN.train_model(dataloader)
myCNN.test_on_train()


test_files = os.listdir("../data/test/unlabeled")
test_files.sort()
n = len(test_files)
labels =["red_blood_cell"]*n


new_transform = transforms.Compose([
    transforms.Resize((52, 52)),
    transforms.ToTensor(), #convert to a tensor
])
new_test_dataset = ImageFolder(root='../data/test', transform=new_transform)
#1372 because thats the number of images in the test dataset
test_loader  = DataLoader(new_test_dataset, batch_size=1372, shuffle=False)
Yhat = None
for i, (X, Y) in enumerate(test_loader): # load a batch data of images
    print(Y)
    P = myCNN.forward(X)  #Compute predicted probabilities
    _, Yhat = torch.max(P, 1)          #Compute predictions
print(Yhat.shape)
labels = []
num_to_label = {0: "gametocyte", 1: "leukocyte", 2: "red_blood_cell", 3: "ring", 4: "schizont", 5: "trophozoite"}
for guess in Yhat:
    number_guess = guess.item()
    labels.append(num_to_label[number_guess])
    

df = pd.DataFrame({"Input": test_files, "Class":labels})
df.to_csv("cnn_predictions.csv",index=False)