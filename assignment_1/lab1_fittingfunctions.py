import numpy as np
# your perceptron implementation here

"""## 1.2 Multilayer Perceptron

Multilayer Perceptron is a fully connected class of neural networks which presents multiple perceptrons connected in a computational graph that "flows" from inputs to outputs.


# 2. How can we build a MLP network in PyTorch?
"""

import torch
import numpy as np
from torch import nn

class LineNetwork(nn.Module):
  # Initialization
  def __init__(self):
    super().__init__()
    # a single perceptron
    self.layers = nn.Sequential(
        nn.Linear(1, 1)
    )

  # how the network operates
  def forward(self, x):
    return self.layers(x)

"""# 3. How can we train a neural network?

Let's train our network to regress a simple affine function. A single Perceptron should be able to approximate a line, right?

## 3.1 Preparing the data infrastructure
"""

from torch.utils.data import Dataset, DataLoader
import torch.distributions.uniform as urand

# This is a Dataset class to work with PyTorch
class AlgebraicDataset(Dataset):
  '''Abstraction for a dataset of a 1D function'''

  def __init__(self, f, interval, nsamples):
    X = urand.Uniform(interval[0], interval[1]).sample([nsamples])
    self.data = [(x, f(x)) for x in X]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

# we need a function
line = lambda x: 2*x + 3
# a domain for our function
interval = (-10, 10)
# the number of points we are going to sample for training
train_nsamples = 1000
# the number of points we are going to evaluate our model on
test_nsamples = 100

train_dataset = AlgebraicDataset(line, interval, train_nsamples)
test_dataset = AlgebraicDataset(line, interval, test_nsamples)

train_dataloader = DataLoader(train_dataset, batch_size=train_nsamples, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_nsamples, shuffle=True)

"""# 3.2 Hyperparameters for Optimization"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"running on {device}")

model = LineNetwork().to(device)

# Loss Function: Mean Squared Error
lossfunc = nn.MSELoss()
# SGD = Stochastic Gradient Descent
# lr = learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

m = nn.Linear(20, 30)
input = torch.randn(128, 9)
#output = model(input)
net = torch.nn.Linear(9,1)
output = net(input)

total_params = sum(p.numel() for p in m.parameters())

print(output.size())


def train(model, dataloader, lossfunc, optimizer):
  '''A function for training our model'''

  model.train()
  cumloss = 0.0
  count = 0
  for X, y in dataloader:
    X = X.unsqueeze(1).float().to(device)
    y = y.unsqueeze(1).float().to(device)

    pred = model(X)
    loss = lossfunc(pred, y)

    # we need to "clean" the accumulated gradients
    optimizer.zero_grad()
    # computes gradients
    loss.backward()
    # updates parameters going in the direction that decreases the local error
    optimizer.step()

    # loss is a tensor so we use *item* to get the underlying float value
    cumloss += loss.item() 
    count += 1
  
  return cumloss / len(dataloader)


def test(model, dataloader, lossfunc):
  '''A function for evaluating our model on test data'''
  model.eval()
  
  cumloss = 0.0
  with torch.no_grad():
    for X, y in dataloader:
      X = X.unsqueeze(1).float().to(device)
      y = y.unsqueeze(1).float().to(device)

      pred = model(X)
      loss = lossfunc(pred, y)
      cumloss += loss.item() 
  
  return cumloss / len(dataloader)

"""# 3.3 Training the network


"""

# for visualization
import imageio
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

# To visualize results
def plot_comparison(f, model, interval=(-10, 10), nsamples=10, return_array=True):
  fig, ax = plt.subplots(figsize=(10, 10))

  ax.grid(True, which='both')
  ax.spines['left'].set_position('zero')
  ax.spines['right'].set_color('none')
  ax.spines['bottom'].set_position('zero')
  ax.spines['top'].set_color('none')

  samples = np.linspace(interval[0], interval[1], nsamples)
  model.eval()
  with torch.no_grad():
    pred = model(torch.tensor(samples).unsqueeze(1).float().to(device))

  ax.plot(samples, list(map(f, samples)), "o", label="ground truth")
  ax.plot(samples, pred.cpu(), label="model")
  plt.legend()
  plt.show()
  # to return image as numpy array
  if return_array:
    fig.canvas.draw()
    img_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return img_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

epochs = 1001

# Let''s make a Gif of the training
filename_output = "./line_approximation.gif"
writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

for t in range(epochs):
  train_loss = train(model, train_dataloader, lossfunc, optimizer)
  if t % 25 == 0:
    print(f"Epoch: {t}; Train Loss: {train_loss}")
    image = plot_comparison(line, model)
    # appending to gif
    writer.append_data(image)

test_loss = test(model, test_dataloader, lossfunc)
print(f"Test Loss: {test_loss}")
writer.close()


class MultiLayerNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(1, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

  def forward(self, x):
    return self.layers(x)

# Write your solution below this cell