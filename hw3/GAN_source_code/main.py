from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from Model import Discriminator, Generator

try:
    os.mkdir("./best_model")
    os.mkdir("./generated")
except:
    pass

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

ngpu = 1
batch_size = 128
lr = 0.0002
beta1 = 0.5
num_epochs = 5

device = torch.device("cuda:0" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")


class celebaDataset(Dataset):
    def __init__(self):
        self.files = os.listdir(os.getcwd()+'/img_align_celeba')
        image_size = 64
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    def __getitem__(self, index):
        image = Image.open(os.getcwd()+'/img_align_celeba/' +
                           self.files[index]).convert('RGB')
        image = self.transform(image)

        return image

    def __len__(self):
        return len(self.files)


def common_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', default='data/celeba', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)

    return parser


def train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, num_epochs):
    # Each epoch, we have to go through every data in dataset
    errg_track = np.full(10, np.inf)
    real_label = 1
    fake_label = 0
    losses_g = []
    losses_d = []
    nz = 100
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    iteration = 0

    for epoch in range(num_epochs):
        # Each iteration, we will get a batch data for training
        for i, data in enumerate(dataloader, 0):

            discriminator.zero_grad()
            # Format batch
            real_img = data.to(device)
            b_size = real_img.size(0)
            label = torch.full((b_size,), real_label, device=device)

            # Forward pass real batch through D
            output = discriminator(real_img).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)

            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)

            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake

            # Update D
            optimizer_d.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizer_g.step()

            losses_g.append(errG.item())
            losses_d.append(errD.item())

            # Use this function to output training procedure while training
            # You can also use this function to save models and samples after fixed number of iteration
            if iteration % 50 == 0:
                if errG.item() < np.max(errg_track):
                    errg_track[np.argmax(errg_track)] = errG.item()
                    save_model(errG.item(), errD.item(), iteration, generator)
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, iteration, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                # plt.figure(figsize=(10, 5))
                # plt.title("Generator and Discriminator Loss During Training")
                # plt.plot(losses_g, label="G")
                # plt.plot(losses_d, label="D")
                # plt.xlabel("iterations")
                # plt.ylabel("Loss")
                # plt.legend()
                # plt.savefig('./loss.png')
                # plt.close()

            if iteration % 500 == 0:
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                    imgs = vutils.make_grid(fake, padding=2, normalize=True)
                    imgs = imgs.numpy()
                    imgs = np.transpose(imgs, (1, 2, 0))
                    imgs = Image.fromarray((imgs * 255).astype(np.uint8))
                    imgs.save('generated/%d.png' % iteration)

            iteration += 1

    # Remember to save all things you need after all batches finished!!!
    with open('loss_g.npy', 'wb') as f:
        np.save(f, losses_g)
    with open('loss_d.npy', 'wb') as f:
        np.save(f, losses_d)

            


def save_model(loss_g, loss_d, iteration, model, max_to_keep=10):
    save_dir = './best_model/'
    model_name = 'lg%.4f_ld%.4f_iter%d.pth' % (loss_g, loss_d, iteration)
    torch.save(model.state_dict(), save_dir+model_name)
    print("save model: %s" % model_name)

    files = os.listdir(save_dir)
    if len(files) > max_to_keep:
        files.sort()
        files = files[::-1]
        os.remove(save_dir+'/'+files[0])


def main(args):
    # Create the dataset by using ImageFolder(get extra point by using customized dataset)
    # remember to preprocess the image by using functions in pytorch
    dataset = celebaDataset()
    # Create the dataloader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)

    # Create the generator and the discriminator()
    # Initialize them
    # Send them to your device
    generator = Generator(ngpu).to(device)
    discriminator = Discriminator(ngpu).to(device)

    # Setup optimizers for both G and D and setup criterion at the same time
    optimizer_g = optim.Adam(generator.parameters(),
                             lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(),
                             lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss()

    # Start training~~

    train(dataloader, generator, discriminator, optimizer_g,
          optimizer_d, criterion, num_epochs)


if __name__ == '__main__':
    args = common_arg_parser()
    main(args)
