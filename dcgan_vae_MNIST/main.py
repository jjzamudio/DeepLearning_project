from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
#import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets
from torchvision.utils import save_image


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=28)
parser.add_argument('--ndf', type=int, default=28)
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--vae_epochs', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

kwargs = {'num_workers': 1, 'pin_memory': True} if opt.cuda else {}

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#
#if opt.dataset in ['imagenet', 'folder', 'lfw']:
#    # folder dataset
#    dataset = dset.ImageFolder(root=opt.dataroot,
#                               transform=transforms.Compose([
#                                   transforms.Resize(opt.imageSize),
#                                   transforms.CenterCrop(opt.imageSize),
#                                   transforms.ToTensor(),
#                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                               ]))
#elif opt.dataset == 'lsun':
#    dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
#                        transform=transforms.Compose([
#                            transforms.Resize(opt.imageSize),
#                            transforms.CenterCrop(opt.imageSize),
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                        ]))
#elif opt.dataset == 'cifar10':
#    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
#                           transform=transforms.Compose([
#                               transforms.Resize(opt.imageSize),
#                               transforms.ToTensor(),
#
#                           ]))
#elif opt.dataset == 'fake':
#    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
#                            transform=transforms.ToTensor())
#assert dataset
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
#                                         shuffle=True, num_workers=int(opt.workers))

train_loader = torch.utils.data.DataLoader(
                                           datasets.MNIST('../data', train=True, download=True,
                                                          transform=transforms.ToTensor()),
                                           batch_size=opt.batchSize, shuffle=True, **kwargs)
pretrain_loader = torch.utils.data.DataLoader(
                                          datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor()),
                                          batch_size=opt.batchSize, shuffle=True, **kwargs)


ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 14 x 14
            nn.ConvTranspose2d(ngf,nc, 2, 2, 0, bias=False),
#            nn.BatchNorm2d(ngf),
#            nn.ReLU(True),
            # state size. (ngf) x  x 32
#            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            #nn.Tanh()
            nn.Sigmoid()
            # state size. (nc) x 28 x 28
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 2, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
#            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ndf * 2),
#            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                                  
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


class _VAE(nn.Module):
    def __init__(self):
        super(_VAE, self).__init__()
        #self.ngpu = ngpu
        
        # for image 3x64x64
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
    
    #for image 3x32x32
    #        self.fc1 = nn.Linear(3072, 1000)
    #        self.fc21 = nn.Linear(1000, 100)
    #        self.fc22 = nn.Linear(1000, 100)
    #        self.fc3 = nn.Linear(100, 1000)
    #        self.fc4 = nn.Linear(1000, 3072)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        #eps = torch.randn_like(std)
        eps = Variable(torch.randn(std.size()))
        if opt.cuda:
            eps = eps.cuda()
        return eps.mul(std).add_(mu)
    
    #        if self.training:
    #            std = torch.exp(0.5*logvar)
    #            #eps = torch.randn_like(std)
    #            eps = Variable(torch.randn(std.size()))
    #            if opt.cuda:
    #                eps = eps.cuda()
    #            return eps.mul(std).add_(mu)
    #        else:
    #            return mu
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))
    #return F.tanh(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    #BCE = criterion(recon_x, x.view(-1, 12288))
    #BCE = criterion(Variable(recon_x.data.resize_(opt.batchSize, 3, opt.imageSize, opt.imageSize)), x)
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD


VAE = _VAE()
print(VAE)
optimizer = optim.Adam(VAE.parameters(), lr=1e-3)
input_vae = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)

if opt.cuda:
    VAE.cuda()
    input_vae = input_vae.cuda()

for epoch in range(1, opt.vae_epochs + 1):
    for i, data in enumerate(pretrain_loader, 0):

        VAE.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input_vae.resize_as_(real_cpu).copy_(real_cpu)
        #input_vae = input/255 # divide by 255 so pixels are in [0,1] range like ouput of autoencoder
        input_vae_v = Variable(input_vae)

        recon_batch, mu, logvar = VAE(input_vae_v)
        loss = loss_function(recon_batch, input_vae_v, mu, logvar)
        loss.backward()
        optimizer.step()

        print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, opt.vae_epochs, i, len(pretrain_loader),
                 loss.data[0]/len(data) ))



criterion = nn.BCELoss() #nn.BCEWithLogitsLoss more numerically stable

input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

D_G_list = []
G_losses = []

for epoch in range(opt.niter):
    for i, data in enumerate(train_loader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
       
       # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
#        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
#        noisev = Variable(noise)
        mu, logvar = VAE.encode(Variable(inputv.data.view(-1, 784)))
        mu.data.resize_(batch_size, nz, 1, 1)
        std = torch.exp(0.5*logvar)
        std.data.resize_(batch_size, nz, 1, 1)
        samples = torch.normal(mu.data, std.data)
        #noise.resize_(batch_size, nz, 1, 1).normal_(mu.data, std.data)
        noisev = Variable(samples)
    
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(train_loader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

        if i % 5 == 0:
            D_G_list.append(D_G_z1)
            G_losses.append(errG.data[0])
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf)
                    #fake = netG(fixed_noise)
            s = torch.normal(mu.data, std.data)
            nv = Variable(s)
            fake = netG(nv)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch))

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

#Save final images for t-SNE
import numpy as np
mydata= (fake.data).cpu().numpy()
#print(mydata.shape)
#(16, 3, 64, 64)
mydata=np.array([mydata[i].flatten() for i in range(mydata.shape[0])])
np.savetxt('%s/fake_images%d_vae_gan.csv' % (opt.outf, epoch),mydata,delimiter=",")

#Real images
mydata=(real_cpu).cpu().numpy()
mydata=np.array([mydata[i].flatten() for i in range(mydata.shape[0])])
np.savetxt('%s/real_images%d_vae_gan.csv' % (opt.outf, epoch),mydata,delimiter=",")

#save D_G
D_G_array = np.array(D_G_list)
np.savetxt('%s/D_G_z%d_vae_gan.csv' % (opt.outf, epoch),D_G_array,delimiter=",")

G_losses_array = np.array(G_losses)
np.savetxt('%s/G_losses%d_vae_gan.csv' % (opt.outf, epoch),G_losses_array,delimiter=",")
