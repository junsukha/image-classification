import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from hw5_utils import BASE_URL, download, GANDataset


class DNet(nn.Module):
    """This is discriminator network."""

    def __init__(self):
        super(DNet, self).__init__()
        
        # TODO: implement layers here
        
        
        self.conv2_1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_2 = nn.Conv2d(2, 4, 3, 1, 1)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(2, 2)
        self.conv2_3 = nn.Conv2d(4, 8, 3, 1, 0)
        self.relu_3 =  nn.ReLU()
        self.fc = nn.Linear(200, 1)

        self._weight_init()


    def _weight_init(self):
        i = 1
        for module in self.children():
#             print(module.get_weight() = True)
#             if module != nn.ReLU():
            if i != 2 and i != 5 and i != 3 and i != 6 and i!=8:
                # print(module)
#             module.weight = nn.init.kaiming_uniform()
                nn.init.kaiming_uniform_(module.weight)
                module.bias.data.fill_(0.0)
            i+=1
 
    def forward(self, x):
        # TODO: complete forward function
#         x.view(x.size(0), 784)
        # print(x.shape)
        # print((self.maxpool_2(self.relu_2(self.conv2_2(self.maxpool_1(self.relu_1(self.conv2_1(x))))))).shape)
        temp = self.relu_3(self.conv2_3(self.maxpool_2(self.relu_2(self.conv2_2(self.maxpool_1(self.relu_1(self.conv2_1(x))))))))
        # print("temp: ", temp.shape)
        # print(temp.view(temp.size(0), -1).shape)
        return self.fc(temp.view(temp.size(0), -1))
#         return self.fc(nn.flatten(self.conv2_3(self.maxpool_2(self.relu_2(self.conv2_2(self.maxpool_1(self.relu_1(self.conv2_1(x)))))))))


class GNet(nn.Module):
    """This is generator network."""

    def __init__(self, zdim):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        super(GNet, self).__init__()

        # TODO: implement layers here
        self.linear = nn.Linear(zdim,1568)
        self.LeakyReLu = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv2d = nn.Conv2d(32,16, 3,1,1)
        self.LeakyReLu2 = nn.LeakyReLU(0.2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv2d2 = nn.Conv2d(16,8,3,1,1)
        self.LeakyReLu3 = nn.LeakyReLU(0.2)
        self.conv2d3 = nn.Conv2d(8, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self._weight_init()

    def _weight_init(self):
        # TODO: implement weight initialization here
        i = 1
        for module in self.children():
#             print(module.weight)
#             if (module != ReLU()):
#             module.weight = nn.init.kaiming_uniform()
            if i != 2 and i!=3 and i != 5 and i!=6 and i != 8 and i != 10:  #use..  if isinstance(child, (nn.Linear, nn.Conv2d))
                nn.init.kaiming_uniform_(module.weight)
                module.bias.data.fill_(0.0)
            i+=1

    def forward(self, z):
        """
        Parameters
        ----------
            z: latent variables used to generate images.
        """
        # TODO: complete forward function
        temp = self.LeakyReLu(self.linear(z))
        return self.sigmoid(self.conv2d3(self.LeakyReLu3(self.conv2d2(self.upsample2(self.LeakyReLu2(self.conv2d(self.upsample(temp.view(-1,32,7,7)))))))))

class GAN:
    def __init__(self, zdim=64):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        torch.manual_seed(2)
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._zdim = zdim
        self.disc = DNet().to(self._dev)
        self.gen = GNet(self._zdim).to(self._dev)

    def _get_loss_d(self, batch_size, batch_data, z):
        """This function computes loss for discriminator.

        Parameters
        ----------
            batch_size: #data per batch.
            batch_data: data from dataset.
            z: random latent variable.
        """
        # TODO: implement discriminator's loss function
        loss_d = 0
        # print(batch_data.shape)
        # print("size", batch_size)
        criterion = nn.BCEWithLogitsLoss()
        # for i in range (batch_size):
        # print(torch.zeros(batch_data.shape).shape)
        # print(self.disc(self.gen(z)).shape)
        zeros = torch.zeros(batch_size , 1).to(self._dev)
        loss_d += criterion( self.disc(self.gen(z)), zeros) #z.shape[0] is # of z's.  z shape: 5 * 64  #compare generated image vs zeros || compare real image vs ones
                                                                                    # only a zero for each data. not for each element of data.
        ones = torch.ones(batch_size , 1).to(self._dev)
        loss_d += criterion(self.disc(batch_data), ones)    #order of parameters matter?!
            # loss_d += criterion(batch_data[i,:,:,:],self.disc(self.gen(z.reshape(z.shape[0],1,8,8)))) #z.shape[0] is # of z's.  z shape: 5 * 64
        return loss_d/2

    def _get_loss_g(self, batch_size, z):
        """This function computes loss for generator.

        Parameters
        ----------
            batch_size: #data per batch.
            z: random latent variable.
        """
        # TODO: implement generator's loss function
        # print(batch_size)
        # print(z.shape)
        # print(self._zdim)
        loss_g = 0
        criterion = nn.BCEWithLogitsLoss()

        ones = torch.ones(batch_size , 1).to(self._dev)
        # for i in range (batch_size):
        loss_g += criterion(self.disc(self.gen(z)), ones)   #order of parameters matter?!
            # loss_g += criterion(batch_data[i,:],self.disc(z.reshape(z.shape[0],1,8,8)))     #z.shape[0] is # of z's.  z shape: 5 * 64    
        return loss_g

    def train(self, iter_d=1, iter_g=1, n_epochs=100, batch_size=256, lr=0.0002):

        # first download
        f_name = "train-images-idx3-ubyte.gz"
        download(BASE_URL + f_name, f_name)

        print("Processing dataset ...")
        train_data = GANDataset(
            f"./data/{f_name}",
            self._dev,
            transform=transforms.Compose([transforms.Normalize((0.0,), (255.0,))]),
        )
        print(f"... done. Total {len(train_data)} data entries.")

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        dopt = optim.Adam(self.disc.parameters(), lr=lr, weight_decay=0.0)
        dopt.zero_grad()
        gopt = optim.Adam(self.gen.parameters(), lr=lr, weight_decay=0.0)
        gopt.zero_grad()

        for epoch in tqdm(range(n_epochs)):
            for batch_idx, data in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):

                z = 2 * torch.rand(data.size()[0], self._zdim, device=self._dev) - 1

                if batch_idx == 0 and epoch == 0:
                    plt.imshow(data[0, 0, :, :].detach().cpu().numpy())
                    plt.savefig("goal.pdf")

                if batch_idx == 0 and epoch % 10 == 0:
                    with torch.no_grad():
                        tmpimg = self.gen(z)[0:64, :, :, :].detach().cpu()
                    save_image(
                        tmpimg, "test_{0}.png".format(epoch), nrow=8, normalize=True
                    )

                dopt.zero_grad()
                for k in range(iter_d):
                    loss_d = self._get_loss_d(batch_size, data, z)
                    loss_d.backward()
                    dopt.step()
                    dopt.zero_grad()

                gopt.zero_grad()
                for k in range(iter_g):
                    loss_g = self._get_loss_g(batch_size, z)
                    loss_g.backward()
                    gopt.step()
                    gopt.zero_grad()

            print(f"E: {epoch}; DLoss: {loss_d.item()}; GLoss: {loss_g.item()}")


if __name__ == "__main__":
    gan = GAN()
    gan.train()
