# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from SaVer_Toolbox import plotter, signedDistanceFunction, verify
from cnn_utils import denorm,generate_noisy_samples_from_image,img_show,logit_samples
from verif_utils import define_classification_polytope_w_b

# %%
betaDKW = 0.001
epsilonDKW = 0.01
Delta = 1-0.99
verifDKW = verify.usingDKW(betaDKW,epsilonDKW,Delta)
betaScenario = 0.001
verifScenario = verify.usingScenario(betaScenario,Delta)

# %%
pretrained_model = "./examples/imageClassification/lenet_mnist_model.pth"

# Set random seed for reproducibility
torch.manual_seed(42)
# LeNet Model definition
class cnnLeNet(nn.Module):
    def __init__(self):
        super(cnnLeNet, self).__init__()
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

# %%
# MNIST dataset and transform
mnist_dataset = datasets.MNIST(root='./examples/imageClassification/data', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ]))

# %%
# Initialize the network
model = cnnLeNet()
# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, weights_only=True,map_location=torch.device('cpu')))
# Set the model in evaluation mode. In this case this is for the Dropout layers: 
model.eval()

# %%
#load normalized image
norm_image, label = mnist_dataset[0]
norm_image = norm_image.unsqueeze(0)
print(f"Original digit: {label}")
print("---------------------------------")

# %%
#Denormalize image:
mean = torch.tensor([0.1307])
std = torch.tensor([0.3081])
image = denorm(norm_image,mean=mean, std=std)
img_show(image,save_image = True,save_name = './examples/imageClassification/original_7.png')

# %%
#define transformation to be applied as required by nn
trans = transforms.Normalize((0.1307,), (0.3081,))
#standard deviation of noise
sd = 0.5
#generate noisy transformed images and corresponding nn output logits
noisySamplesDKW,noisyLogitsDKW = logit_samples(image,model,trans,verifDKW.samplesRequired(),sd)
noisySamplesScenario,noisyLogitsScenario = logit_samples(image,model,trans,verifScenario.samplesRequired(),sd)
img_show(denorm(noisySamplesDKW[0],mean=mean, std=std),save_image = True,save_name = './examples/imageClassification/noisy7DKW.png')
img_show(denorm(noisySamplesScenario[0],mean=mean, std=std),save_image = True,save_name = './examples/imageClassification/noisy7Scenario.png')

# %%
#Defining polytope for given class index
wDKW,bDKW = define_classification_polytope_w_b(noisyLogitsDKW[0],label)
sdfDKW = signedDistanceFunction.polytope(wDKW,bDKW)
wScenario,bScenario = define_classification_polytope_w_b(noisyLogitsScenario[0],label)
sdfScenario = signedDistanceFunction.polytope(wScenario,bScenario)
verifDKW.specification(sdfDKW)
verifScenario.specification(sdfScenario)

# %%
# Add the noisy samples to the verification problem: 
verifDKW.samples(noisyLogitsDKW)
verifScenario.samples(noisyLogitsScenario)

# %%
# Check if the samples satisfy the specification: 
verifDKW.probability()
verifScenario.probability()


