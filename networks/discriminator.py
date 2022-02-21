from torch import nn

    
class get_xone_discriminator(nn.Module):
    def __init__(self, img_ch=64):
        super(get_xone_discriminator, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(img_ch, img_ch, kernel_size=4, stride=2, padding=1),#128
        nn.BatchNorm2d(img_ch),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(img_ch, img_ch, kernel_size=4, stride=2, padding=1),#64
        nn.BatchNorm2d(img_ch),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(img_ch, img_ch,  kernel_size=4, stride=2, padding=1),#32
        nn.BatchNorm2d(img_ch),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(img_ch, img_ch, kernel_size=4, stride=2, padding=1),#16
        nn.BatchNorm2d(img_ch),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(img_ch, img_ch, kernel_size=4, stride=2, padding=1),#8
        nn.BatchNorm2d(img_ch),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(img_ch, img_ch, kernel_size=4, stride=2, padding=1), #4      
        nn.BatchNorm2d(img_ch),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(img_ch, 1, kernel_size=4, stride=1, padding=0), #1 
    )
        
#        self.active1=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc1 = nn.Linear(512, 1)
#        self.active2=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = x.view(x.size(0), -1)
#        x=self.active1(x)
#        x=self.fc1(x)
#        x=self.active2(x)
#        x=self.fc2(x)
#        output =self.sigmoid(x)

        return x

class get_xtwo_discriminator(nn.Module):
    def __init__(self, img_ch=128):
        super(get_xtwo_discriminator, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(img_ch, 128, kernel_size=4, stride=2, padding=1),#128
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),#64
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(128, 128,  kernel_size=4, stride=2, padding=1),#32
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),#16
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),#8
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0), #4      
    )
        
#        self.active1=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc1 = nn.Linear(512, 1)
#        self.active2=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = x.view(x.size(0), -1)
#        x=self.active1(x)
#        x=self.fc1(x)
#        x=self.active2(x)
#        x=self.fc2(x)
#        output =self.sigmoid(x)

        return x


class get_xfive_discriminator(nn.Module):
    def __init__(self, img_ch=1024):
        super(get_xfive_discriminator, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(img_ch, img_ch*2, kernel_size=4, stride=2, padding=1),  #input 16 16 output  8 8
        nn.BatchNorm2d(img_ch*2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(img_ch*2, img_ch*2, kernel_size=4, stride=2, padding=1), #input  8 8 output  4 4
        nn.BatchNorm2d(img_ch*2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(img_ch*2, img_ch*4, kernel_size=4, stride=2, padding=1), #input  4 4 output  2 2
        nn.BatchNorm2d(img_ch*4),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(img_ch*4, img_ch*4, kernel_size=2, stride=1, padding=0), #input  2 2 output  1 1
        nn.BatchNorm2d(img_ch*4),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(img_ch*4, img_ch*4, kernel_size=1, stride=1, padding=0), #input  2 2 output  1 1 
        nn.BatchNorm2d(img_ch*4),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(img_ch*4, img_ch*4, kernel_size=1, stride=1, padding=0), #input  2 2 output  1 1 
        nn.BatchNorm2d(img_ch*4),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(img_ch*4, 1, kernel_size=1, stride=1, padding=0), #input  2 2 output  1 1 
    )
        
#        self.active1=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc1 = nn.Linear(128, 64)
#        self.active2=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = x.view(x.size(0), -1)
#        x=self.active1(x)
#        x=self.fc1(x)
#        x=self.active2(x)
#        x=self.fc2(x)
#        output =self.sigmoid(x)

        return x


class get_dtwo_discriminator(nn.Module):
    def __init__(self, maps=64):
        super(get_dtwo_discriminator, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(maps, maps* 2, kernel_size=4, stride=2, padding=1), #128
        nn.BatchNorm2d(maps*2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(maps* 2, maps * 2, kernel_size=4, stride=2, padding=1),#64
        nn.BatchNorm2d(maps*2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(maps * 2, maps * 2, kernel_size=4, stride=2, padding=1),#32
        nn.BatchNorm2d(maps*2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(maps * 2, maps, kernel_size=4, stride=2, padding=1),#16
        nn.BatchNorm2d(maps),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(maps, maps, kernel_size=4, stride=2, padding=1),#8
        nn.BatchNorm2d(maps),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(maps, maps, kernel_size=4, stride=2, padding=1),# 4
        nn.BatchNorm2d(maps),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(maps , 1, kernel_size=4, stride=1, padding=0),# 1
    )
        
#        self.active1=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc1 = nn.Linear(128, 64)
#        self.active2=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = x.view(x.size(0), -1)
#        x=self.active1(x)
#        x=self.fc1(x)
#        x=self.active2(x)
#        x=self.fc2(x)
#        output =self.sigmoid(x)

        return x



class get_done_discriminator(nn.Module):
    def __init__(self, img_ch=1):
        super(get_done_discriminator, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(img_ch, 16, kernel_size=4, stride=2, padding=1),#128
        nn.BatchNorm2d(16),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),#64
        nn.BatchNorm2d(32),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),#32
        nn.BatchNorm2d(32),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),#16
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),#8
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #4      
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0), #1 
    )
        
#        self.active1=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc1 = nn.Linear(128, 64)
#        self.active2=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = x.view(x.size(0), -1)
#        x=self.active1(x)
#        x=self.fc1(x)
#        x=self.active2(x)
#        x=self.fc2(x)
#        output =self.sigmoid(x)

        return x

class get_done_entropy_discriminator(nn.Module):
    def __init__(self, img_ch=1):
        super(get_done_entropy_discriminator, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(img_ch, 16, kernel_size=4, stride=2, padding=1),#128
        nn.BatchNorm2d(16),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),#64
        nn.BatchNorm2d(32),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),#32
        nn.BatchNorm2d(32),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),#16
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),#8
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #4      
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0), #1 
    )
        
#        self.active1=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc1 = nn.Linear(128, 64)
#        self.active2=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = x.view(x.size(0), -1)
#        x=self.active1(x)
#        x=self.fc1(x)
#        x=self.active2(x)
#        x=self.fc2(x)
#        output =self.sigmoid(x)

        return x


class get_exit_discriminator(nn.Module):
    def __init__(self, img_ch=1):
        super(get_exit_discriminator, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(img_ch, 16, kernel_size=4, stride=2, padding=1),#128
        nn.BatchNorm2d(16),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),#64
        nn.BatchNorm2d(32),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),#32
        nn.BatchNorm2d(32),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),#16
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),#8
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #4      
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0), #1 
    )
        
#        self.active1=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc1 = nn.Linear(128, 64)
#        self.active2=nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = x.view(x.size(0), -1)
#        x=self.active1(x)
#        x=self.fc1(x)
#        x=self.active2(x)
#        x=self.fc2(x)
#        output =self.sigmoid(x)

        return x


