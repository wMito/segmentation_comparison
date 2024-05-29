from torch import nn
import torch
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = nn.Sequential(
            # input: [1, 912, 1216]
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), 
            ##nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            ##nn.BatchNorm2d(64)
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            ##nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            ##nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
            #nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), 
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512, 114, 152
            #nn.BatchNorm2d(512
            nn.ReLU()
        )
        self.enc5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), 
            #nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), 
            #nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2) # 512, 114, 152
        )
        self.decod1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), # 512, 114, 152
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1), # 512, 114, 152
            #nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 2, stride=2) # 256
        )
        self.decod2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 2, stride=2)
        )
        self.decod3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 2, stride=2)
        )
        self.decod4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1, stride=1)
        )

        

    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        inp1 = torch.cat((e4, e5), 1) # 1024, 114, 152
        d1 = self.decod1(inp1)
        inp2 = torch.cat((e3, d1), 1) 
        d2 = self.decod2(inp2)
        inp3 = torch.cat((e2, d2), 1) 
        d3 = self.decod3(inp3)
        inp4 = torch.cat((e1, d3), 1) 
        d4 = self.decod4(inp4)

        return nn.Sigmoid()(d4)
