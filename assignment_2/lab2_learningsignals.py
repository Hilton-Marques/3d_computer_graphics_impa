import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import imageio

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Grayscale
from torchvision.transforms.functional import to_tensor, to_pil_image
from collections import OrderedDict
import numpy as np
import skimage
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.fft import fft, fftfreq
from typing import Sequence, Tuple, Union

import time

import seaborn as sns

torch.manual_seed(777)
sns.set_theme()

"""# 1 Sampling and Reconstruction

We are going to start working with 1D signals in the interval $[-1, 1]$. Shannon theorem states some conditions for reconstruction of bandlimited signals after uniform sampling. Let's verify it in practice.
"""

def get_mgrid(sidelen, dim=2, start=-1, end=1):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(start, end, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class Signal1D(Dataset):
    def __init__(self, coords, values):
        super().__init__()
        self.values = values.view(-1, 1)
        self.coords = coords

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
        return self.coords, self.values

def sinusoidal(nsamples, freqs, biases=None):
    coords = get_mgrid(nsamples, 1).view(-1)
    values = torch.zeros(nsamples)
    if biases is None:
      biases = torch.zeros(len(freqs))
    for freq, bias in zip(freqs, biases):
      values = values + torch.sin(freq * coords + bias)
    return Signal1D(coords, values)

def plot_signals(signals:Union[Signal1D, Sequence[Signal1D]]):
    COLORS = ['blue', 'orange', 'green', 'purple', 'pink']
    lines = ['-', '--', '-.', ':']
    fig, ax = plt.subplots(figsize=(16, 8))
    if not isinstance(signals, Sequence):
      signals = [signals]
    for i, signal in enumerate(signals):
        ax.plot(signal.coords, 
                signal.values, 
                label=f'S{i}', color=COLORS[i%len(COLORS)])
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')

    # remove the ticks from the top and right edges
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.legend()

"""## 1.1 Uniform Sampling

Let's take some samples of continuous functions uniformly over the interval $[-1, 1]$ an plot them. To make it easier to see we are working with bandlimited signals, we are going to start with combinations of sinusoidal functions, so we know the exact the frequency content of the signal.
"""

def plot_samples1D(func, nsamples):
    x = get_mgrid(nsamples, 1)
    samples = func(x)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x, samples, 'o', label=f'Uniform samples')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    # remove the ticks from the top and right edges
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.legend()

def multi_plot(dots, curves=[], labels=[]):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(*dots, 'o', label=f'Uniform samples')
    for i, (x, y) in enumerate(curves):
        ax.plot(x, y, label=labels[i])

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    # remove the ticks from the top and right edges
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.legend()

gridres = 1025
nsamples = 91
signal = lambda x: (np.cos(2 * x) + np.sin(4 * x))
full_grid = get_mgrid(gridres, 1)
samples_grid = get_mgrid(nsamples, 1)

multi_plot((samples_grid, signal(samples_grid)), 
           [(full_grid, signal(full_grid))], 
           ['original'])

"""## 1.2 Reconstruction

* Implement a function to reconstruct a signal sampled uniformly using the Shanon basis, that is the function $sinc(x) = \frac{sin(x)}{x}$. Plot the original signal and the reconstructed signal (sample more points) in the same graph. Did it work? Why?

* Reduce the sampling rate until it's below the Nyquist limit. What happens when you try to reconstruct the signal? Alternatively, you may add higher frequencies to the signal.

* Try again with a signal composed of more frequencies.
"""

# Suggested template
def shannon_reconstruction(x, y):
    # compute what you need
    eps = 1e-6
    sinc = lambda x: (torch.sin( torch.pi * x) / ( torch.pi * x + eps))
    n_samples = x.size(0)
    f_s = 1/(x[1] - x[0]) #sampling frequency
    res = 0.0
    def rec(t):
        n_grid = t.size(0)
        a = torch.kron(torch.ones(1 , n_samples),t).transpose(0,1)
        b = torch.kron(x, torch.ones(1, n_grid))
        return torch.mm(y.transpose(0,1) , torch.sinc(f_s * (a - b))).transpose(0,1)
    # return a function that given a coordinate t, 
    # computes the reconstructed value
    return rec

# write your answer here
apr_signal = shannon_reconstruction(samples_grid,signal(samples_grid))
res = apr_signal(full_grid)

multi_plot((samples_grid, signal(samples_grid)), 
           [(full_grid, signal(full_grid)), (full_grid, res)], 
           ['original' , 'reconstruction'])

rec_signal_obj = Signal1D(full_grid, apr_signal(full_grid))


a = 1

"""## 1.3 Fourier Transform

The code below allows you to plot the Fast Fourier Transform of a signal, and check its representation in the frequency domain.
"""

def plot_fft1D(signals, color_order = None):
    COLORS = ['blue', 'orange', 'green', 'purple', 'pink']
    if not color_order:
      color_order = [0,1,2,3,4]
    lines = ['-', '--', '-.', ':']
    if not isinstance(signals, Sequence):
      signals = [signals]
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, signal in enumerate(signals):
        W = signal.values.view(-1).cpu().detach().numpy()
        N = len(W)
        yf = fft(W)
        xf = fftfreq(N, 2/N)[:N//2]
        ax.plot(xf, 2.0/N * np.abs(yf[0:N//2]),'o',
                label=f'S{i}', color=COLORS[color_order[i]%len(color_order)])
    plt.xlim([0,5])
    plt.legend()

"""*   Based on the experiments you did on the previous section, use the FFT algorithm to compare the frequencies present in the original signal and its reconstruction. Check it when you have a sampling rate above the Nyquist limit, and also when it's below this limit. Do you see anything interesting?"""

# write your answer here
plot_fft1D([rec_signal_obj],[0,2])


"""# 2 Aliasing

We saw that Aliasing is an effect that causes higher frequencies to be interpreted as lower frequencies under certain conditions. We are going to check how aliasing artifacts look like when we are working with 1D and 2D signals.

![image.png](data:image/png;base64,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)

## 2.1 Spoting aliasing artifacts

Perlin Noise is a kind of stochastic signal with large applications in procedural modeling in computer graphics. We are going to use it to generate more sophisticated signals for the next exercise.
"""

def blend(x):
  return 6*x**5 - 15*x**4 + 10*x**3

def noise(scale, samples):
    # create a list of 2d vectors
    angles = torch.rand(scale) * 2*torch.pi
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

    x = torch.linspace(0, scale-1, samples)
    noise_values = []
    for value in x[:-1]:
        i = torch.floor(value).long()
        lower, upper = gradients[i], gradients[i+1]

        dot1 = torch.dot(lower, torch.tensor([value - i, 0]))
        dot2 = torch.dot(upper, torch.tensor([value - i - 1, 0]))
        # TODO: review interpolation
        k1 = blend(value-i)
        k2 = blend(i+1 - value)
        interpolated = k1*dot2 + k2*dot1 
        noise_values.append(interpolated.item())
    
    noise_values.append(0.0)
    return torch.tensor(noise_values)

def perlin_noise(nsamples, scale=10, octaves=1, p=1):
    pnoise = 0
    for i in range(octaves):
        partial = noise(2**i * scale, nsamples)/(p**i)
        pnoise = partial + pnoise
    return Signal1D(get_mgrid(nsamples, 1), pnoise)

"""* Generate some Perlin Noise signals exploring different setting of its parameters. When you have an idea of how it works, choose one that presents fine details. Plot the signal and its FFT. Analyze what you see."""

# write your answer here

n_samples = gridres
perlin = perlin_noise(65)
perlin_high = perlin_noise(n_samples)

samples_perlin = get_mgrid(65, 1)
rec_perlin = shannon_reconstruction(samples_perlin, perlin.values)
multi_plot([], 
           [(full_grid, perlin_high.values),(full_grid, rec_perlin(full_grid))], 
           ['original','reconstruction'])

"""* Create a new Perlin Noise signal from your previously generated signal by naively subsampling it: just keep 1 sample for every *sumbsampling_factor* samples and ignore the others.
* Use the sinc basis to reconstruct the noise signal and compare the original signal to the reconstructed signal. Can you spot any alising artifacts? If so, describe them.
* Check the FFT plot for both the original noise signal and the reconstructed noise signal.
"""

# write your answer here

"""## 2.2 Aliasing Artifacts in Images

Download the image of a striped t-shirt using the code below, then load and display the image.
"""

# download image

tshirt = Image.open('camisa.jpg')
tshirt

"""* Convert the image to a tensor or numpy array representation and naively subsample it by a factor of 8. After checking the result, can you see any alising artifacts? Explain.

* Try a higher subsampling factor. Does it display more artifacts?

If you prefer, you can resize the image to a bigger size using the following snippet:

```python
# img is a numpy array that contains the pixels values
# dim is the new size; e.g. 513
Image.fromarray(img).resize((dim, dim), Image.BICUBIC)
```
"""

# write your answer here
sampling_factor = 2

tshirt_np = np.array(tshirt)
tshirt_torch = torch.from_numpy(tshirt_np)
tshirt_size = tshirt_torch.size()

new_size_x = tshirt_size[0]//sampling_factor 
new_size_y = tshirt_size[1]//sampling_factor 

mask_x = torch.zeros(1,tshirt_size[0], dtype=torch.bool)
mask_y = torch.zeros(tshirt_size[1],1, dtype=torch.bool)

mask_x[0,0:-1:sampling_factor] = True
mask_y[0:-1:sampling_factor,0] = True

mask = torch.kron(mask_x,mask_y)
sampled_img = tshirt_torch[mask,:].reshape((-1,new_size_y,3)).float()

id_prev = tshirt_size[1]//2

signal = tshirt_torch[:, id_prev, 0]

id_curr = new_size_y//2

samples_grid_2 = get_mgrid(new_size_y, 1)

signal_sub_sampling = sampled_img[:, id_curr, 0].reshape((new_size_y,1)).float()
rec_signal = shannon_reconstruction(samples_grid_2, signal_sub_sampling)



multi_plot([], 
           [ (full_grid , signal), (full_grid, rec_signal(full_grid))], 
           ['original', 'reconstruction'])

"""# 3 Fitting an 2D signal (image) using a neural network

In the previous assignment, you learned how to train a neural network to fit a 1D function using the basic components in the PyTorch framework. Now, we are going to solve a similar problem, but in 2D.


"""

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())        
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.0]), torch.Tensor([1.0]))
    ])
    img = transform(img)
    return img

def get_tensor(img_np, sidelength):
    img = Image.fromarray(img_np)        
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.0]), torch.Tensor([1.0]))
    ])
    img = transform(img)
    return img

class ImageSignal(Dataset):
    def __init__(self, sidelength, path=None):
        super().__init__()
        if path is None:
          img = get_cameraman_tensor(sidelength)
        else:
          img = Image.open(path).convert('L').resize((sidelength, sidelength))
          img = to_tensor(img)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)
        self.dim = sidelength

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
        return self.coords, self.pixels
    
class ImageSignalRGB(Dataset):
    def __init__(self, sidelength, gray_scale = True, path=None):
        super().__init__()
        if gray_scale:
            img = Image.open(path).convert('L').resize((sidelength, sidelength))
            img = to_tensor(img)
            self.pixels = img.permute(1, 2, 0).view(-1, 1)
        else:
            img = Image.open(path).resize((sidelength, sidelength))
            img = to_tensor(img)
            self.pixels = img.permute(1, 2, 0).view(-1, 3)
        self.coords = get_mgrid(sidelength, 2)
        self.dim = sidelength

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
        return self.coords, self.pixels

class ReLuNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, coords):
        output = self.layers(coords)
        return output

def plot_tensor_img(output, dim, return_img=False):
    img = output.cpu().view(dim, dim, -1).detach().numpy()
    if return_img:
        return img
    plt.grid(False)
    plt.imshow(img, cmap='gray')

def train(model, dataloader, epochs, device, steps_til_summary=100, gif_path=""):
    model.to(device)
    model.train()
    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.to(device), ground_truth.to(device)
    
    dim = int(np.sqrt(ground_truth.size(0)))
    if gif_path:
        writer = imageio.get_writer(gif_path, mode='I', duration=0.3)

    for step in range(epochs):
        model_output = model(model_input)    
        loss = ((model_output - ground_truth)**2).mean()

        if not (step % steps_til_summary):
            print("Step %d, Total loss %0.6f" % (step, loss))
            plot_tensor_img(model_output, dim)

        if gif_path and (step % 5 == 0):
            img = np.clip(model_output.cpu().view(dim, dim,-1).detach().numpy(), 0, 1) 
            writer.append_data(np.uint8(img * 255))

        optim.zero_grad()
        loss.backward()
        optim.step()
    # last inference
    model.eval()
    model_output = model(model_input)
    img = np.clip(plot_tensor_img(model_output, dim, True), 0, 1)
    writer.append_data(np.uint8(img * 255))
    writer.close()

"""## 3.1 Training a regular MLP using ReLu as activation function

* Train a model using the ReLuNetwork provided in code cell above. Try to fit at least 2 images. What do you observe in the result?

* **[Optional]** Can you do better? Experiment different activation functions and/or a different set of hyperparameters and try to improve the result. Feel free to change the network architecture.


"""
example = ImageSignalRGB(256,gray_scale=False, path="./my_cats.jpeg")

model = ReLuNetwork()
train(model, example, 2, "cpu" , gif_path="./teste.gif")
# write your answer here

"""## 3.2 Fourier Transform of Images"""

def plot_fft2D(pixels:torch.Tensor, dim):
    pixels_trans = pixels.view(dim, dim, -1).squeeze(-1).permute(2, 0, 1)
    transform = Grayscale()
    pixels_trans = transform(pixels_trans).permute(1, 2, 0)
    fourier_tensor = torch.fft.fftshift(
                    torch.fft.fft2(pixels.view(dim, dim).squeeze(-1)))
    magnitude = 20 * np.log(abs(fourier_tensor.numpy()) + 1e-10)
    mmin = np.min(magnitude)
    magnitude = (magnitude - mmin) / (np.max(magnitude) - mmin)
    img = np.uint8(magnitude * 255)
    plt.grid(False)
    plt.imshow(img, cmap='gray')


plot_fft2D(example.pixels, example.dim)

"""The code above computes the Fast Fourier Transform of a 2D image and displays the magnitude of the FFT as another image. The lower frequencies will appear near the center of the image while the higher frequencies will appear near the border of the image. The magnitude is normalized for better visualization."""

example = ImageSignal(256)
plot_tensor_img(example.pixels, example.dim)

plot_fft2D(example.pixels, example.dim)

sidelength = 256
coords = get_mgrid(sidelength, 2)

coords.to("cpu")

model.to("cpu")
model.eval()
with torch.no_grad():
    model_output = model(coords)
plot_tensor_img(model_output, sidelength, True)


plot_fft2D(model_output, sidelength)

"""* Visualize the FFT of the result image of the trained model. How does it compare to the ground truth?"""

# write your answer here

"""## 3.3 Training a Sinusoidal Neural Network to fit an image"""

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output

"""The code above defines a simplified version of a SIREN* (sinusoidal representation network. This kind of network uses sines as the activation function. 

Notice that it's not as trivial as changing $relu()$ for $sin()$, as it requires a special initialization to guarantee stability and convergence during training. You can find the paper and the original [code in this link](https://www.vincentsitzmann.com/siren/), and check the details.

* Train a Siren network to fit the same images you were working with. Is the result better? Explain.

* Use the FFT to analyze the frequencies learned by the network. Why do you think 
"""

# write your answer here