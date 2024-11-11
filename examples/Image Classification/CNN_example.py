#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from cnn_utils import denorm,generate_noisy_samples_from_image,img_show,logit_samples
from verif_utils import define_classification_polytope_w_b,generate_A_b,signed_distance_function


# In[2]:


from nnSampleVerification import sdfs, verifMethods, plotter


# In[3]:


pretrained_model = "lenet_mnist_model.pth"
use_cuda=True

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


# In[4]:


# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# In[5]:


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# In[6]:


# MNIST dataset and transform
mnist_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ]))


# In[7]:


# Initialize the network
model = Net().to(device)
# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location=device, weights_only=True))
# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()


# In[8]:


#load normalized image
norm_image, label = mnist_dataset[0]
norm_image = norm_image.unsqueeze(0)


# In[9]:


if label == 7:
    print("The chosen image is of digit 7")
else:
    print("Choose another image i.e. of digit 7 from MNIST dataset")


# In[10]:


#denormalize image
mean = torch.tensor([0.1307]).to(device)
std = torch.tensor([0.3081]).to(device)
image = denorm(norm_image,mean=mean, std=std)


# In[11]:


#define transformation to be applied as required by nn
trans = transforms.Normalize((0.1307,), (0.3081,))


# In[12]:


#standard deviation of noise
sd = 0.5


# In[13]:


alpha = 0.001
epsilon = 0.1
verif1 = verifMethods.dkw(epsilon,alpha)


# In[14]:


#generate noisy transformed images and corresponding nn output logits
noisy_samples,noisy_logits = logit_samples(image,model,trans,verif1.num_samples,sd)


# In[15]:


verif1.addSamples(noisy_logits)


# In[16]:


#Defining polytope for given class index
W,B = define_classification_polytope_w_b(noisy_logits[0],label)


# In[17]:


class NN_SDF:
    def __init__(self,W,B):
        self.W = W
        self.B = B

    def reviseCenter(self,W,B):
        self.W = W
        self.B = B

    def eval(self, point, zero_radius):
        # Replace this with your actual signed distance function implementation
        eval = np.array([signed_distance_function(l,self.W,self.B) for l in point]) - zero_radius 
        return eval


# In[18]:


SDF = NN_SDF(W,B)
verif1.addSpecification(SDF)


# In[19]:


tol = 1e-12


# In[20]:


# Run ZeroOne
verif1.findZeroOne()


# In[21]:


levelProb = 0.99
levelSetZeroRadius = verif1.findLevelSet(levelProb)


# In[22]:


print(f"We need a SDF of {levelSetZeroRadius} to satisfy the specification with the desired probability\n")
if levelSetZeroRadius > 0:
    print(f"Since the required SDF is greater than 0, we do not satisfy the specification")
else:
    print(f"Since the required SDF is lesser than 0, we satisfy the specification")

