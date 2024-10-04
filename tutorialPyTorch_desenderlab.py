# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:56:53 2024

@author: helen
"""

#load packages
import torch                                     
from torch.utils.data import DataLoader          
from torchvision import datasets                 
from torchvision.transforms import ToTensor      
import matplotlib.pyplot as plt                  
from torch import nn                             

#load data 
training_data = datasets.MNIST(
    root='data', 
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

#show example of the data
labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

#preparing data for training
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size= batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size= batch_size, shuffle=True)

#create model structure
class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 10 , kernel_size = 3, stride = 1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
            )
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10*12*12, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        logits = self.linear_relu_stack(x)
        return logits

#choose the model that you want to use, by commenting out the other one
model = SimpleNeuralNetwork()
#model = ConvNeuralNetwork()
print(model)

#visualize model architecture
import torchviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/helen/Documents/KULeuven/master 2/stage/tutorials programmeren/Graphviz-12.1.2-win32/bin'

dummy_input = torch.randn(1, 1, 28, 28)  # MNIST images are 1x28x28
output = model(dummy_input)

torchviz.make_dot(output, params=dict(model.named_parameters())).render("model_graph", format="png")
#it works, but it is not a nice visualization 

#define functions for training and evaluation of the model
epochs = 3
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_dataloader.dataset) for i in range(epochs + 1)]

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        #track the losses for visualization
        if batch % 10 == 0:
            train_losses.append(loss.item())
            train_counter.append((batch*64) + (epoch*len(train_dataloader.dataset)))

        #track progress of the training
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    test_losses.append(test_loss)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#train and evaluate the mode
learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss() #this step applies the softmax transformation and can compare the output of the model with integers
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

test_loop(test_dataloader, model, loss_fn)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, t)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

#create a graph
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(train_counter, train_losses, color='cornflowerblue', linewidth=1.5)
plt.scatter(test_counter, test_losses, zorder = 5, color='darkred')         
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig

#visualize model architecture
import torchviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/helen/Documents/KULeuven/master 2/stage/tutorials programmeren/Graphviz-12.1.2-win32/bin'


torchviz.make_dot(model, params=dict(model.named_parameters())).render("model_graph", format="png")
#it works, but it is not a nice visualization 

#save  model
torch.save(model, 'modelMNIST.pth')

#if you would want to load the model again
model = torch.load('modelMNIST.pth', weights_only=False), 

