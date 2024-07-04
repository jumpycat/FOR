import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.in1 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)

        # # Down-sampling layers.
        curr_dim = conv_dim
        self.conv2 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(curr_dim*2, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.in3 = nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)

        # # Bottleneck layers.
        curr_dim = curr_dim * 2
        self.resblk1 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.resblk2 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.resblk3 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.resblk4 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.resblk5 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.resblk6 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)

        # # Up-sampling layers.
        self.upconv1 = nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.in4 = nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)
        self.upconv2 = nn.ConvTranspose2d(curr_dim//2, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.in5 = nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)

        curr_dim = curr_dim // 2
        self.conv4 = nn.Conv2d(curr_dim, 3 + 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()
        


        self.my_module_conv1 = nn.Conv2d(curr_dim * 2, curr_dim * 2, kernel_size=3, stride=2)
        self.my_module_conv2 = nn.Conv2d(curr_dim * 2, curr_dim * 2, kernel_size=3, stride=2)
        self.my_module_conv3 = nn.Conv2d(curr_dim * 2, 2048, kernel_size=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.my_module_fc1 = nn.Linear(2048,512)
        self.my_module_fc2 = nn.Linear(512,512)
        self.my_module_fc3 = nn.Linear(512,curr_dim * 2)
        self.sigmoid = nn.Sigmoid()

        self.my_conv1 = nn.Conv2d(curr_dim * 2, curr_dim * 2, kernel_size=3, stride=1, padding=1)

        # layers = []
        # layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        # layers.append(nn.ReLU(inplace=True))

        # # Down-sampling layers.
        # curr_dim = conv_dim
        # for i in range(2):
        #     layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        #     layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
        #     layers.append(nn.ReLU(inplace=True))
        #     curr_dim = curr_dim * 2

        # # Bottleneck layers.
        # for i in range(repeat_num):
        #     layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # # Up-sampling layers.
        # for i in range(2):
        #     layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
        #     layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
        #     layers.append(nn.ReLU(inplace=True))
        #     curr_dim = curr_dim // 2

        # layers.append(nn.Conv2d(curr_dim, 3+1, kernel_size=7, stride=1, padding=3, bias=False))
        # # layers.append(nn.Tanh()) # ht
        # self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        input_image = x
        x = torch.cat([x, c], dim=1)
        
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.in3(x)
        x = self.relu(x)

        x = self.resblk1(x)
        x = self.resblk2(x)
        x = self.resblk3(x)
        x = self.resblk4(x)
        x = self.resblk5(x)
        x = self.resblk6(x)
        

        ######   my_module ######
        x2 = self.my_module_conv1(x)
        x2 = self.my_module_conv2(x2)
        x2 = self.my_module_conv3(x2)
        x2 = self.avgpool(x2)

        x3 = x2.view(x2.size(0), -1)
        x3 = self.my_module_fc1(x3)
        x3 = self.my_module_fc2(x3)
        x3 = self.my_module_fc3(x3)
        x3 = torch.unsqueeze(torch.unsqueeze(x3,-1),-1)
        x3 = self.sigmoid(x3)

        x = self.my_conv1(x)
        x = x3 * x 
        ######   my_module ######


        x = self.upconv1(x)
        x = self.in4(x)
        x = self.relu(x)  

        x = self.upconv2(x)
        x = self.in5(x)
        x = self.relu(x)  
 
        output = self.conv4(x)
        
        
        # x = self.tanh(x)
        # return x, x2, x3
    
        # output = self.main(x)
        # print(output.size())
        attention_mask = F.sigmoid(output[:, :1])
        content_mask = output[:, 1:]
        attention_mask = attention_mask.repeat(1, 3, 1, 1)
        # print(content_mask.shape)
        # print(attention_mask.shape)
        # print(input_image.shape)

        result = content_mask * attention_mask + input_image * (1 - attention_mask)
        return result, attention_mask, content_mask,x2,x3


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
