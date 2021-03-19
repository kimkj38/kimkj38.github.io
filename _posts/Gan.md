```
from torchvision import datasets
import torchvision.transforms as transforms
import torch

#number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 64

# convert data to torch.Floattensor
transform = transforms.ToTensor()

# get the training datasets
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)

# prepare data loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
```


```
import numpy as np
import matplotlib.pyplot as plt


# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

#get one image from the batch
img = np.squeeze(images[0])

fig = plt.figure(figsize = (3, 3))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7fb5f4a11c90>




    
![png](Gan_files/Gan_1_1.png)
    



```
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
  
  def __init__(self, input_size, hidden_dim, output_size):
    super(Discriminator, self).__init__()

    # define hidden linear layers
    self.fc1 = nn.Linear(input_size, hidden_dim*4)
    self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
    self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)

    # final fully-connected layers
    self.fc4 = nn.Linear(hidden_dim, output_size)

    # dropout layer
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    # flatten image
    x = x.view(-1, 28*28)
    # all hidden layers
    x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
    x = self.dropout(x)
    x = F.leaky_relu(self.fc2(x), 0.2)
    x = self.dropout(x)
    x = F.leaky_relu(self.fc3(x), 0.2)
    x = self.dropout(x)
    # final layer
    out = self.fc4(x)

    return out

```


```
class Generator(nn.Module):

  def __init__(self, input_size, hidden_dim, output_size):
    super(Generator, self).__init__()

    # define hidden linear layers
    self.fc1 = nn.Linear(input_size, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
    self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)

    # final fully-connected layer
    self.fc4 = nn.Linear(hidden_dim*4, output_size)

    # dropout layer
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    # all hidden layers
    x = F.leaky_relu(self.fc1(x), 0.2)
    x = self.dropout(x)
    x = F.leaky_relu(self.fc2(x), 0.2)
    x = self.dropout(x)
    x = F.leaky_relu(self.fc3(x), 0.2)
    x = self.dropout(x)
    #final layer with tanh applied
    out = F.tanh(self.fc4(x))

    return out
```


```
# Discriminator hyperparams

# Size of input image to discriminator (28*28)
input_size = 784
# Size of discriminator output (real or fake)
d_output_size = 1
# Size of last hidden layer in the discriminator
d_hidden_size = 32

# Generator hyperparams

# Size of latent vector to give to generator
z_size = 100
# Size of discriminator output (generated image)
g_output_size = 784
# Size of first hidden layer in the generator
g_hidden_size = 32
# instantiate discriminator and generator
D = Discriminator(input_size, d_hidden_size, d_output_size)
G = Generator(z_size, g_hidden_size, g_output_size)

# check that they are as you expect
print(D)
print()
print(G)
```

    Discriminator(
      (fc1): Linear(in_features=784, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=64, bias=True)
      (fc3): Linear(in_features=64, out_features=32, bias=True)
      (fc4): Linear(in_features=32, out_features=1, bias=True)
      (dropout): Dropout(p=0.3, inplace=False)
    )
    
    Generator(
      (fc1): Linear(in_features=100, out_features=32, bias=True)
      (fc2): Linear(in_features=32, out_features=64, bias=True)
      (fc3): Linear(in_features=64, out_features=128, bias=True)
      (fc4): Linear(in_features=128, out_features=784, bias=True)
      (dropout): Dropout(p=0.3, inplace=False)
    )



```
# Calculate losses
def real_loss(D_out, smooth=False):
  batch_size = D_out.size(0)
  # label smoothing
  if smooth:
    # smooth, real labels = 0.9
    labels = torch.ones(batch_size) * 0.9
  else:
    labels = torch.ones(batch_size) 
  
  # numerically stable loss
  criterion = nn.BCEWithLogitsLoss()
  # calculate loss
  loss = criterion(D_out.squeeze(), labels)
  return loss

def fake_loss(D_out):
  batch_size = D_out.size(0)
  labels = torch.zeros(batch_size)
  criterion = nn.BCEWithLogitsLoss()
  # calculate loss
  loss = criterion(D_out.squeeze(), labels)
  return loss

import torch.optim as optim

# Optimizers
lr = 0.002

# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr)
g_optimizer = optim.Adam(G.parameters(), lr)
```


```
import pickle as pkl

# training hyperparams
num_epochs = 100

# keep track of loss and generated, "fake" samples
samples = []
losses = []

print_every = 400

# Get som fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size=16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()

#train the network
D.train()
G.train()
for epoch in range(num_epochs):
  for batch_i, (real_images, _) in enumerate(train_loader):
    batch_size = real_images.size(0)

    ## important rescaling step ##
    real_images = real_images*2 - 1 #rescale input images from [0,1] to [-1,1]

    # =========================================
    #         Train the Discriminator
    # =========================================

    d_optimizer.zero_grad()

    # 1. Train with real images

    # compute the discriminator losses on real images
    # smooth the real labels
    D_real = D(real_images)
    d_real_loss = real_loss(D_real, smooth=True)

    # 2. Train with fake images

    # Generate fake images
    z = np.random.uniform(-1, 1, size=(batch_size, z_size))
    z = torch.from_numpy(z).float()
    fake_images = G(z)

    # compute the discriminator losses on fake images
    d_fake = D(fake_images)
    d_fake_loss = fake_loss(d_fake)

    #add up loss and perform backprop
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    d_optimizer.step()

    # ============================================
    #            Train the Generator
    # ============================================
    g_optimizer.zero_grad()


    # 1. Train with fake images and flipped labels

    # Generate fake images
    z = np.random.uniform(-1, 1, size=(batch_size, z_size))
    z = torch.from_numpy(z).float()
    fake_images = G(z)

    # compute the discriminate losses on fake images
    # using flipped labels!
    D_fake = D(fake_images)
    g_loss = real_loss(D_fake)

    #perfrom backprop
    g_loss.backward()
    g_optimizer.step()

    #print some loss stats
    if batch_i%print_every == 0:
      # print discriminator and generator loss
      print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
          epoch+1, num_epochs, d_loss.item(), g_loss.item()
      ))

  ## After each epoch ##
  # append discriminator loss and generator loss
  losses.append((d_loss.item(), g_loss.item()))

  #generate and save sample, fake images
  G.eval() # eval mode for generating samples
  samples_z = G(fixed_z)
  samples.append(samples_z)
  G.train()


```

    /usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:1628: UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.
      warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")


    Epoch [    1/  100] | d_loss: 1.4551 | g_loss: 0.5765
    Epoch [    1/  100] | d_loss: 0.7764 | g_loss: 2.2614
    Epoch [    1/  100] | d_loss: 1.1250 | g_loss: 1.1340
    Epoch [    2/  100] | d_loss: 0.9071 | g_loss: 1.7795
    Epoch [    2/  100] | d_loss: 0.9064 | g_loss: 1.6545
    Epoch [    2/  100] | d_loss: 1.2521 | g_loss: 1.4581
    Epoch [    3/  100] | d_loss: 1.1508 | g_loss: 2.7541
    Epoch [    3/  100] | d_loss: 0.9997 | g_loss: 1.9692
    Epoch [    3/  100] | d_loss: 1.0578 | g_loss: 1.3832
    Epoch [    4/  100] | d_loss: 1.4213 | g_loss: 1.2034
    Epoch [    4/  100] | d_loss: 1.1885 | g_loss: 1.4492
    Epoch [    4/  100] | d_loss: 1.1958 | g_loss: 1.3885
    Epoch [    5/  100] | d_loss: 1.3251 | g_loss: 1.3154
    Epoch [    5/  100] | d_loss: 1.2741 | g_loss: 0.9596
    Epoch [    5/  100] | d_loss: 1.2243 | g_loss: 1.4967
    Epoch [    6/  100] | d_loss: 1.1575 | g_loss: 1.0996
    Epoch [    6/  100] | d_loss: 1.2725 | g_loss: 1.1856
    Epoch [    6/  100] | d_loss: 1.1835 | g_loss: 1.2210
    Epoch [    7/  100] | d_loss: 1.2490 | g_loss: 0.8683
    Epoch [    7/  100] | d_loss: 1.1125 | g_loss: 1.4043
    Epoch [    7/  100] | d_loss: 1.0840 | g_loss: 1.5359
    Epoch [    8/  100] | d_loss: 1.1366 | g_loss: 1.7684
    Epoch [    8/  100] | d_loss: 1.2248 | g_loss: 1.2851
    Epoch [    8/  100] | d_loss: 1.2395 | g_loss: 0.9324
    Epoch [    9/  100] | d_loss: 1.2748 | g_loss: 1.6341
    Epoch [    9/  100] | d_loss: 1.0183 | g_loss: 1.2788
    Epoch [    9/  100] | d_loss: 1.1579 | g_loss: 1.4864
    Epoch [   10/  100] | d_loss: 1.1693 | g_loss: 1.2563
    Epoch [   10/  100] | d_loss: 1.2036 | g_loss: 0.9115
    Epoch [   10/  100] | d_loss: 1.2524 | g_loss: 1.0076
    Epoch [   11/  100] | d_loss: 1.2811 | g_loss: 1.4529
    Epoch [   11/  100] | d_loss: 1.2398 | g_loss: 1.3534
    Epoch [   11/  100] | d_loss: 1.1933 | g_loss: 1.0814
    Epoch [   12/  100] | d_loss: 1.3567 | g_loss: 0.8987
    Epoch [   12/  100] | d_loss: 1.1507 | g_loss: 1.4445
    Epoch [   12/  100] | d_loss: 1.2902 | g_loss: 1.1040
    Epoch [   13/  100] | d_loss: 1.2226 | g_loss: 1.3552
    Epoch [   13/  100] | d_loss: 1.0885 | g_loss: 1.3077
    Epoch [   13/  100] | d_loss: 1.2501 | g_loss: 1.1701
    Epoch [   14/  100] | d_loss: 1.2856 | g_loss: 1.0963
    Epoch [   14/  100] | d_loss: 1.2093 | g_loss: 1.3232
    Epoch [   14/  100] | d_loss: 1.3356 | g_loss: 1.1160
    Epoch [   15/  100] | d_loss: 1.3475 | g_loss: 1.2689
    Epoch [   15/  100] | d_loss: 1.2076 | g_loss: 1.1436
    Epoch [   15/  100] | d_loss: 1.3976 | g_loss: 1.0605
    Epoch [   16/  100] | d_loss: 1.2689 | g_loss: 0.8262
    Epoch [   16/  100] | d_loss: 1.2505 | g_loss: 1.0416
    Epoch [   16/  100] | d_loss: 1.2801 | g_loss: 1.1447
    Epoch [   17/  100] | d_loss: 1.3533 | g_loss: 1.3715
    Epoch [   17/  100] | d_loss: 1.2999 | g_loss: 1.4325
    Epoch [   17/  100] | d_loss: 1.2410 | g_loss: 1.1567
    Epoch [   18/  100] | d_loss: 1.2654 | g_loss: 1.0355
    Epoch [   18/  100] | d_loss: 1.2412 | g_loss: 0.9251
    Epoch [   18/  100] | d_loss: 1.3622 | g_loss: 1.0871
    Epoch [   19/  100] | d_loss: 1.3296 | g_loss: 0.7685
    Epoch [   19/  100] | d_loss: 1.2741 | g_loss: 0.9580
    Epoch [   19/  100] | d_loss: 1.2653 | g_loss: 1.1157
    Epoch [   20/  100] | d_loss: 1.2777 | g_loss: 0.9267
    Epoch [   20/  100] | d_loss: 1.2907 | g_loss: 0.9018
    Epoch [   20/  100] | d_loss: 1.3281 | g_loss: 0.9144
    Epoch [   21/  100] | d_loss: 1.2354 | g_loss: 1.2966
    Epoch [   21/  100] | d_loss: 1.2673 | g_loss: 0.8138
    Epoch [   21/  100] | d_loss: 1.2694 | g_loss: 1.0642
    Epoch [   22/  100] | d_loss: 1.3301 | g_loss: 1.0520
    Epoch [   22/  100] | d_loss: 1.2628 | g_loss: 1.0041
    Epoch [   22/  100] | d_loss: 1.2173 | g_loss: 1.1477
    Epoch [   23/  100] | d_loss: 1.2589 | g_loss: 0.9596
    Epoch [   23/  100] | d_loss: 1.2205 | g_loss: 1.1126
    Epoch [   23/  100] | d_loss: 1.3601 | g_loss: 1.0074
    Epoch [   24/  100] | d_loss: 1.2076 | g_loss: 1.2259
    Epoch [   24/  100] | d_loss: 1.2871 | g_loss: 0.9384
    Epoch [   24/  100] | d_loss: 1.2485 | g_loss: 1.2904
    Epoch [   25/  100] | d_loss: 1.2847 | g_loss: 1.1638
    Epoch [   25/  100] | d_loss: 1.2102 | g_loss: 1.0303
    Epoch [   25/  100] | d_loss: 1.2077 | g_loss: 1.3980
    Epoch [   26/  100] | d_loss: 1.2629 | g_loss: 1.2026
    Epoch [   26/  100] | d_loss: 1.1987 | g_loss: 1.0215
    Epoch [   26/  100] | d_loss: 1.3709 | g_loss: 1.0458
    Epoch [   27/  100] | d_loss: 1.3979 | g_loss: 1.1454
    Epoch [   27/  100] | d_loss: 1.1487 | g_loss: 1.1839
    Epoch [   27/  100] | d_loss: 1.1403 | g_loss: 1.4121
    Epoch [   28/  100] | d_loss: 1.3053 | g_loss: 1.0394
    Epoch [   28/  100] | d_loss: 1.2781 | g_loss: 1.0214
    Epoch [   28/  100] | d_loss: 1.2665 | g_loss: 0.9528
    Epoch [   29/  100] | d_loss: 1.2446 | g_loss: 1.2201
    Epoch [   29/  100] | d_loss: 1.2703 | g_loss: 0.9209
    Epoch [   29/  100] | d_loss: 1.2402 | g_loss: 0.8468
    Epoch [   30/  100] | d_loss: 1.2231 | g_loss: 1.4218
    Epoch [   30/  100] | d_loss: 1.2639 | g_loss: 1.0039
    Epoch [   30/  100] | d_loss: 1.4204 | g_loss: 0.9561
    Epoch [   31/  100] | d_loss: 1.2354 | g_loss: 0.9835
    Epoch [   31/  100] | d_loss: 1.2316 | g_loss: 1.0331
    Epoch [   31/  100] | d_loss: 1.4370 | g_loss: 0.9094
    Epoch [   32/  100] | d_loss: 1.2998 | g_loss: 0.9710
    Epoch [   32/  100] | d_loss: 1.3879 | g_loss: 0.9507
    Epoch [   32/  100] | d_loss: 1.4506 | g_loss: 1.1304
    Epoch [   33/  100] | d_loss: 1.2157 | g_loss: 0.9542
    Epoch [   33/  100] | d_loss: 1.3346 | g_loss: 1.0152
    Epoch [   33/  100] | d_loss: 1.2663 | g_loss: 0.9653
    Epoch [   34/  100] | d_loss: 1.1955 | g_loss: 1.6679
    Epoch [   34/  100] | d_loss: 1.3537 | g_loss: 0.9782
    Epoch [   34/  100] | d_loss: 1.3027 | g_loss: 1.1217
    Epoch [   35/  100] | d_loss: 1.3141 | g_loss: 0.9064
    Epoch [   35/  100] | d_loss: 1.3158 | g_loss: 1.2531
    Epoch [   35/  100] | d_loss: 1.3012 | g_loss: 0.8580
    Epoch [   36/  100] | d_loss: 1.2851 | g_loss: 0.8750
    Epoch [   36/  100] | d_loss: 1.2698 | g_loss: 0.8875
    Epoch [   36/  100] | d_loss: 1.3541 | g_loss: 1.0294
    Epoch [   37/  100] | d_loss: 1.2725 | g_loss: 1.0593
    Epoch [   37/  100] | d_loss: 1.2189 | g_loss: 0.9837
    Epoch [   37/  100] | d_loss: 1.2863 | g_loss: 0.9955
    Epoch [   38/  100] | d_loss: 1.3151 | g_loss: 0.9658
    Epoch [   38/  100] | d_loss: 1.3766 | g_loss: 0.9813
    Epoch [   38/  100] | d_loss: 1.4720 | g_loss: 1.0329
    Epoch [   39/  100] | d_loss: 1.3128 | g_loss: 0.9834
    Epoch [   39/  100] | d_loss: 1.2434 | g_loss: 0.9628
    Epoch [   39/  100] | d_loss: 1.4134 | g_loss: 1.0993
    Epoch [   40/  100] | d_loss: 1.3847 | g_loss: 1.0314
    Epoch [   40/  100] | d_loss: 1.1641 | g_loss: 1.0287
    Epoch [   40/  100] | d_loss: 1.3305 | g_loss: 0.8790
    Epoch [   41/  100] | d_loss: 1.3875 | g_loss: 1.3014
    Epoch [   41/  100] | d_loss: 1.3214 | g_loss: 0.9699
    Epoch [   41/  100] | d_loss: 1.2877 | g_loss: 1.1453
    Epoch [   42/  100] | d_loss: 1.3334 | g_loss: 0.9345
    Epoch [   42/  100] | d_loss: 1.3812 | g_loss: 0.9441
    Epoch [   42/  100] | d_loss: 1.3023 | g_loss: 0.9742
    Epoch [   43/  100] | d_loss: 1.3589 | g_loss: 1.1044
    Epoch [   43/  100] | d_loss: 1.2780 | g_loss: 1.0490
    Epoch [   43/  100] | d_loss: 1.3342 | g_loss: 0.9960
    Epoch [   44/  100] | d_loss: 1.3008 | g_loss: 1.2360
    Epoch [   44/  100] | d_loss: 1.2456 | g_loss: 0.8597
    Epoch [   44/  100] | d_loss: 1.2995 | g_loss: 1.2296
    Epoch [   45/  100] | d_loss: 1.7105 | g_loss: 1.4483
    Epoch [   45/  100] | d_loss: 1.2591 | g_loss: 0.9695
    Epoch [   45/  100] | d_loss: 1.2894 | g_loss: 0.8508
    Epoch [   46/  100] | d_loss: 1.2434 | g_loss: 1.1023
    Epoch [   46/  100] | d_loss: 1.2605 | g_loss: 1.2027
    Epoch [   46/  100] | d_loss: 1.3069 | g_loss: 0.9025
    Epoch [   47/  100] | d_loss: 1.2871 | g_loss: 1.0729
    Epoch [   47/  100] | d_loss: 1.3167 | g_loss: 1.1815
    Epoch [   47/  100] | d_loss: 1.3725 | g_loss: 1.0180
    Epoch [   48/  100] | d_loss: 1.2679 | g_loss: 1.2854
    Epoch [   48/  100] | d_loss: 1.2989 | g_loss: 1.0268
    Epoch [   48/  100] | d_loss: 1.3280 | g_loss: 0.9615
    Epoch [   49/  100] | d_loss: 1.2875 | g_loss: 0.8311
    Epoch [   49/  100] | d_loss: 1.3339 | g_loss: 1.1403
    Epoch [   49/  100] | d_loss: 1.3299 | g_loss: 1.0448
    Epoch [   50/  100] | d_loss: 1.3485 | g_loss: 1.3001
    Epoch [   50/  100] | d_loss: 1.2470 | g_loss: 0.9125
    Epoch [   50/  100] | d_loss: 1.3137 | g_loss: 1.2346
    Epoch [   51/  100] | d_loss: 1.3521 | g_loss: 1.0353
    Epoch [   51/  100] | d_loss: 1.3406 | g_loss: 0.9564
    Epoch [   51/  100] | d_loss: 1.2793 | g_loss: 1.1363
    Epoch [   52/  100] | d_loss: 1.3015 | g_loss: 1.2908
    Epoch [   52/  100] | d_loss: 1.2103 | g_loss: 1.2020
    Epoch [   52/  100] | d_loss: 1.2506 | g_loss: 1.0161
    Epoch [   53/  100] | d_loss: 1.3296 | g_loss: 1.2671
    Epoch [   53/  100] | d_loss: 1.2564 | g_loss: 1.1508
    Epoch [   53/  100] | d_loss: 1.3842 | g_loss: 1.0265
    Epoch [   54/  100] | d_loss: 1.2748 | g_loss: 2.0645
    Epoch [   54/  100] | d_loss: 1.3259 | g_loss: 0.9663
    Epoch [   54/  100] | d_loss: 1.3765 | g_loss: 1.0881
    Epoch [   55/  100] | d_loss: 1.2617 | g_loss: 1.4100
    Epoch [   55/  100] | d_loss: 1.1240 | g_loss: 0.9173
    Epoch [   55/  100] | d_loss: 1.3436 | g_loss: 1.0675
    Epoch [   56/  100] | d_loss: 1.2408 | g_loss: 0.9075
    Epoch [   56/  100] | d_loss: 1.2985 | g_loss: 0.9671
    Epoch [   56/  100] | d_loss: 1.3862 | g_loss: 0.9708
    Epoch [   57/  100] | d_loss: 1.3291 | g_loss: 0.6941
    Epoch [   57/  100] | d_loss: 1.2067 | g_loss: 1.0530
    Epoch [   57/  100] | d_loss: 1.3234 | g_loss: 1.0045
    Epoch [   58/  100] | d_loss: 1.2511 | g_loss: 0.9531
    Epoch [   58/  100] | d_loss: 1.2900 | g_loss: 0.9874
    Epoch [   58/  100] | d_loss: 1.2529 | g_loss: 0.8053
    Epoch [   59/  100] | d_loss: 1.3174 | g_loss: 1.1379
    Epoch [   59/  100] | d_loss: 1.2718 | g_loss: 0.9362
    Epoch [   59/  100] | d_loss: 1.2769 | g_loss: 1.1814
    Epoch [   60/  100] | d_loss: 1.2953 | g_loss: 1.1370
    Epoch [   60/  100] | d_loss: 1.2624 | g_loss: 0.9837
    Epoch [   60/  100] | d_loss: 1.3184 | g_loss: 0.8831
    Epoch [   61/  100] | d_loss: 1.2370 | g_loss: 1.0283
    Epoch [   61/  100] | d_loss: 1.2838 | g_loss: 1.2199
    Epoch [   61/  100] | d_loss: 1.2998 | g_loss: 1.0187
    Epoch [   62/  100] | d_loss: 1.2900 | g_loss: 1.3272
    Epoch [   62/  100] | d_loss: 1.3426 | g_loss: 1.1074
    Epoch [   62/  100] | d_loss: 1.3040 | g_loss: 0.9586
    Epoch [   63/  100] | d_loss: 1.2736 | g_loss: 1.0398
    Epoch [   63/  100] | d_loss: 1.2783 | g_loss: 0.9545
    Epoch [   63/  100] | d_loss: 1.2831 | g_loss: 0.9884
    Epoch [   64/  100] | d_loss: 1.3651 | g_loss: 1.0318
    Epoch [   64/  100] | d_loss: 1.3327 | g_loss: 1.0482
    Epoch [   64/  100] | d_loss: 1.3543 | g_loss: 0.9656
    Epoch [   65/  100] | d_loss: 1.3183 | g_loss: 0.8280
    Epoch [   65/  100] | d_loss: 1.1782 | g_loss: 0.9140
    Epoch [   65/  100] | d_loss: 1.2897 | g_loss: 0.9239
    Epoch [   66/  100] | d_loss: 1.2477 | g_loss: 1.1170
    Epoch [   66/  100] | d_loss: 1.3283 | g_loss: 0.9581
    Epoch [   66/  100] | d_loss: 1.5061 | g_loss: 1.1437
    Epoch [   67/  100] | d_loss: 1.3484 | g_loss: 1.0182
    Epoch [   67/  100] | d_loss: 1.2586 | g_loss: 1.0100
    Epoch [   67/  100] | d_loss: 1.2399 | g_loss: 1.4547
    Epoch [   68/  100] | d_loss: 1.3434 | g_loss: 1.0001
    Epoch [   68/  100] | d_loss: 1.3332 | g_loss: 0.9957
    Epoch [   68/  100] | d_loss: 1.2472 | g_loss: 1.0110
    Epoch [   69/  100] | d_loss: 1.3281 | g_loss: 1.0159
    Epoch [   69/  100] | d_loss: 1.3077 | g_loss: 1.3234
    Epoch [   69/  100] | d_loss: 1.3075 | g_loss: 1.1193
    Epoch [   70/  100] | d_loss: 1.1649 | g_loss: 0.9343
    Epoch [   70/  100] | d_loss: 1.2308 | g_loss: 1.1710
    Epoch [   70/  100] | d_loss: 1.2812 | g_loss: 1.0634
    Epoch [   71/  100] | d_loss: 1.4751 | g_loss: 1.2454
    Epoch [   71/  100] | d_loss: 1.3281 | g_loss: 0.9392
    Epoch [   71/  100] | d_loss: 1.4033 | g_loss: 0.9979
    Epoch [   72/  100] | d_loss: 1.3203 | g_loss: 0.8976
    Epoch [   72/  100] | d_loss: 1.2338 | g_loss: 0.7834
    Epoch [   72/  100] | d_loss: 1.2798 | g_loss: 0.9328
    Epoch [   73/  100] | d_loss: 1.2990 | g_loss: 0.8441
    Epoch [   73/  100] | d_loss: 1.2219 | g_loss: 1.0707
    Epoch [   73/  100] | d_loss: 1.3172 | g_loss: 1.3033
    Epoch [   74/  100] | d_loss: 1.2570 | g_loss: 1.1545
    Epoch [   74/  100] | d_loss: 1.2490 | g_loss: 0.8509
    Epoch [   74/  100] | d_loss: 1.4852 | g_loss: 0.9869
    Epoch [   75/  100] | d_loss: 1.2323 | g_loss: 1.3871
    Epoch [   75/  100] | d_loss: 1.2421 | g_loss: 0.9632
    Epoch [   75/  100] | d_loss: 1.2957 | g_loss: 0.9343
    Epoch [   76/  100] | d_loss: 1.2565 | g_loss: 1.0173
    Epoch [   76/  100] | d_loss: 1.3084 | g_loss: 1.1266
    Epoch [   76/  100] | d_loss: 1.3887 | g_loss: 1.0247
    Epoch [   77/  100] | d_loss: 1.3235 | g_loss: 0.9395
    Epoch [   77/  100] | d_loss: 1.1812 | g_loss: 1.0858
    Epoch [   77/  100] | d_loss: 1.4035 | g_loss: 1.0818
    Epoch [   78/  100] | d_loss: 1.3207 | g_loss: 0.9730
    Epoch [   78/  100] | d_loss: 1.1787 | g_loss: 1.0153
    Epoch [   78/  100] | d_loss: 1.3236 | g_loss: 1.0790
    Epoch [   79/  100] | d_loss: 1.3350 | g_loss: 1.1306
    Epoch [   79/  100] | d_loss: 1.1127 | g_loss: 1.2721
    Epoch [   79/  100] | d_loss: 1.2825 | g_loss: 0.9523
    Epoch [   80/  100] | d_loss: 1.1914 | g_loss: 1.0580
    Epoch [   80/  100] | d_loss: 1.2216 | g_loss: 1.0701
    Epoch [   80/  100] | d_loss: 1.3073 | g_loss: 0.9969
    Epoch [   81/  100] | d_loss: 1.2962 | g_loss: 0.8621
    Epoch [   81/  100] | d_loss: 1.2105 | g_loss: 1.3372
    Epoch [   81/  100] | d_loss: 1.3519 | g_loss: 0.9065
    Epoch [   82/  100] | d_loss: 1.2925 | g_loss: 1.4523
    Epoch [   82/  100] | d_loss: 1.2652 | g_loss: 1.0121
    Epoch [   82/  100] | d_loss: 1.3527 | g_loss: 0.9022
    Epoch [   83/  100] | d_loss: 1.3230 | g_loss: 0.9695
    Epoch [   83/  100] | d_loss: 1.4202 | g_loss: 0.9843
    Epoch [   83/  100] | d_loss: 1.3540 | g_loss: 1.0898
    Epoch [   84/  100] | d_loss: 1.2746 | g_loss: 1.0591
    Epoch [   84/  100] | d_loss: 1.1762 | g_loss: 1.1784
    Epoch [   84/  100] | d_loss: 1.2466 | g_loss: 0.9990
    Epoch [   85/  100] | d_loss: 1.2744 | g_loss: 1.1387
    Epoch [   85/  100] | d_loss: 1.2537 | g_loss: 0.9885
    Epoch [   85/  100] | d_loss: 1.4212 | g_loss: 1.0741
    Epoch [   86/  100] | d_loss: 1.3333 | g_loss: 0.8710
    Epoch [   86/  100] | d_loss: 1.2547 | g_loss: 1.0381
    Epoch [   86/  100] | d_loss: 1.2330 | g_loss: 1.0631
    Epoch [   87/  100] | d_loss: 1.4568 | g_loss: 1.1991
    Epoch [   87/  100] | d_loss: 1.2120 | g_loss: 1.0856
    Epoch [   87/  100] | d_loss: 1.4051 | g_loss: 1.1011
    Epoch [   88/  100] | d_loss: 1.3295 | g_loss: 0.9963
    Epoch [   88/  100] | d_loss: 1.3605 | g_loss: 0.9660
    Epoch [   88/  100] | d_loss: 1.3436 | g_loss: 0.9536
    Epoch [   89/  100] | d_loss: 1.3148 | g_loss: 1.0444
    Epoch [   89/  100] | d_loss: 1.1960 | g_loss: 1.0615
    Epoch [   89/  100] | d_loss: 1.3155 | g_loss: 0.8727
    Epoch [   90/  100] | d_loss: 1.2684 | g_loss: 1.3693
    Epoch [   90/  100] | d_loss: 1.2945 | g_loss: 0.8122
    Epoch [   90/  100] | d_loss: 1.3322 | g_loss: 0.9933
    Epoch [   91/  100] | d_loss: 1.2756 | g_loss: 1.3320
    Epoch [   91/  100] | d_loss: 1.2394 | g_loss: 1.0574
    Epoch [   91/  100] | d_loss: 1.3594 | g_loss: 1.0661
    Epoch [   92/  100] | d_loss: 1.2607 | g_loss: 1.0214
    Epoch [   92/  100] | d_loss: 1.1941 | g_loss: 1.2326
    Epoch [   92/  100] | d_loss: 1.2606 | g_loss: 0.9415
    Epoch [   93/  100] | d_loss: 1.1886 | g_loss: 1.2483
    Epoch [   93/  100] | d_loss: 1.2828 | g_loss: 1.0234
    Epoch [   93/  100] | d_loss: 1.3865 | g_loss: 1.1544
    Epoch [   94/  100] | d_loss: 1.2750 | g_loss: 1.1818
    Epoch [   94/  100] | d_loss: 1.3278 | g_loss: 0.9290
    Epoch [   94/  100] | d_loss: 1.3800 | g_loss: 1.0034
    Epoch [   95/  100] | d_loss: 1.2802 | g_loss: 0.9520
    Epoch [   95/  100] | d_loss: 1.1734 | g_loss: 0.9732
    Epoch [   95/  100] | d_loss: 1.3113 | g_loss: 1.2095
    Epoch [   96/  100] | d_loss: 1.3719 | g_loss: 1.1403
    Epoch [   96/  100] | d_loss: 1.2598 | g_loss: 1.0454
    Epoch [   96/  100] | d_loss: 1.2605 | g_loss: 1.1610
    Epoch [   97/  100] | d_loss: 1.4885 | g_loss: 1.6393
    Epoch [   97/  100] | d_loss: 1.2792 | g_loss: 1.0769
    Epoch [   97/  100] | d_loss: 1.3619 | g_loss: 0.9078
    Epoch [   98/  100] | d_loss: 1.2473 | g_loss: 1.0104
    Epoch [   98/  100] | d_loss: 1.2526 | g_loss: 1.2067
    Epoch [   98/  100] | d_loss: 1.2944 | g_loss: 1.0712
    Epoch [   99/  100] | d_loss: 1.2983 | g_loss: 0.9700
    Epoch [   99/  100] | d_loss: 1.1682 | g_loss: 1.0402
    Epoch [   99/  100] | d_loss: 1.2849 | g_loss: 1.2149
    Epoch [  100/  100] | d_loss: 1.3284 | g_loss: 0.9449
    Epoch [  100/  100] | d_loss: 1.0971 | g_loss: 1.2272
    Epoch [  100/  100] | d_loss: 1.2723 | g_loss: 1.3376



    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-50-f0088e33b260> in <module>()
         96 # Save training generator samples
         97 with open('train_samples.pkl', 'wb') as f:
    ---> 98   pkl.dump(samples. f)
         99 fig, ax = plt.subplots()
        100 losses = np.array(losses)


    AttributeError: 'list' object has no attribute 'f'



```
# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
  pkl.dump(samples, f)
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fb5f4b29910>




    
![png](Gan_files/Gan_7_1.png)
    



```
# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
  fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
  for ax, img in zip(axes.flatten(), samples[epoch]):
    img = img.detach()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
# Load samples from generator, taken while traing
with open('train_samples.pkl', 'rb') as f:
  samples = pkl.load(f)
```


```
view_samples(epoch, samples)
```


    
![png](Gan_files/Gan_9_0.png)
    



```

```
