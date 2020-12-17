
path = '00001.jpg'
URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/%D0%9F%D0%BE%D1%80%D1%82%D1%80%D0%B5%D1%82%D0%BD%D0%BE%D0%B5_%D1%84%D0%BE%D1%82%D0%BE_%D0%91.%D0%94.%D0%9C%D0%B5%D0%BD%D0%B4%D0%B5%D0%BB%D0%B5%D0%B2%D0%B8%D1%87%D0%B0.jpg/400px-%D0%9F%D0%BE%D1%80%D1%82%D1%80%D0%B5%D1%82%D0%BD%D0%BE%D0%B5_%D1%84%D0%BE%D1%82%D0%BE_%D0%91.%D0%94.%D0%9C%D0%B5%D0%BD%D0%B4%D0%B5%D0%BB%D0%B5%D0%B2%D0%B8%D1%87%D0%B0.jpg'



from pretrainedmodels import dpn92
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pickle
from torchvision.datasets.folder import default_loader
from typing import List
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
from matplotlib import pyplot as plt



class FA(nn.Module):
    def __init__(self, num_ch):
        super().__init__()
        self.Q = nn.Sequential(
            nn.Conv2d(num_ch, 32, 1),
            nn.BatchNorm2d(32)
        )
        self.K = nn.Sequential(
            nn.Conv2d(num_ch, 32, 1),
            nn.BatchNorm2d(32)
        )
        self.V = nn.Sequential(
            nn.Conv2d(num_ch, num_ch, 1),
            nn.BatchNorm2d(num_ch)
        )
        self.convBNrelu = nn.Sequential(
            nn.Conv2d(num_ch, num_ch, 1),
            nn.BatchNorm2d(num_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V0 = self.V(x)

        N,C,H,W = x.size()

        Q_ = Q.view(N,32,-1).permute(0, 2, 1)
        Q = F.normalize(Q_, p=2, dim=2, eps=1e-12)

        K_   = K.view(N,32,-1)
        K   = F.normalize(K_, p=2, dim=1, eps=1e-12)

        V_ = V0.view(N,C,-1).permute(0, 2, 1)
        V = F.relu(V_)

        f = torch.matmul(K, V)
        y = torch.matmul(Q, f)
        y = y.permute(0, 2, 1).contiguous()

        y = y.view(N, C, H, W)
        W = self.convBNrelu(y)
        return W + V0
        
class Upsample(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode=self.mode, align_corners=True)

class FADPNEncoder(nn.Module):
    def __init__(self, num_classes=2, arch="dpn92"):
        super().__init__()
        backbone = dpn92(pretrained='imagenet+5k').features

        self.conv0 = nn.Sequential(backbone[0])
        self.block1 = nn.Sequential(*backbone[1:4])
        self.block2 = nn.Sequential(*backbone[4:8])
        self.block3 = nn.Sequential(*backbone[8:28])
        self.block4 = nn.Sequential(*backbone[28:])

    def forward(self, x):
        acts = []
        x = self.conv0(x)
        for i in range(4):
            x = self.__getattr__(f"block{i + 1}")(x)
            acts.append(x)
        return acts

class FADPNDecoder(nn.Module):    
    def __init__(self):
        super().__init__()   
        self.fa1 = FA(2688)
        self.fa2 = FA(1552)
        self.fa3 = FA(704)
        self.fa4 = FA(336)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(2688, 1552, 1),
            nn.BatchNorm2d(1552),
            Upsample((50, 38), mode='bilinear'),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(1552, 704, 1),
            nn.BatchNorm2d(704),
            Upsample((100, 75), mode='bilinear'),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(704, 336, 1),
            nn.BatchNorm2d(336),
            Upsample((200, 150), mode='bilinear'),
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(336, 1, 1),
            nn.BatchNorm2d(1),
            Upsample((800, 600), mode='bilinear')
        )

    def forward(self, acts):
        x = self.decoder1(self.fa1(acts[3]))
        x = self.decoder2(x + self.fa2(torch.cat(acts[2], dim=1)))
        x = self.decoder3(x + self.fa3(torch.cat(acts[1], dim=1)))
        x = self.decoder4(x + self.fa4(torch.cat(acts[0], dim=1)))
        return x

def segm(encoder, decoder, URL, path):

	# from file
	#img = default_loader(path)

	# from URL
	response = requests.get(URL)
	img = Image.open(BytesIO(response.content))

	old_size = img.size
	to_tensor = transforms.ToTensor()
	img = img.resize([600, 800])
	img = to_tensor(img)
	img = img.view(-1, *(img.shape))


	out = decoder(encoder(img))

	result = Image.fromarray((out[0][0] > 1).numpy())
	result = result.resize(old_size)
	result.save('result.jpg')
	return result


