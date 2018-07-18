import torch
import model.LRN as LRN

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, 11, 4, 7),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),
            LRN.LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, 5, 1, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),
            LRN.LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 256, 3, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.upsample=torch.nn.ConvTranspose2d(256, 256, 16, 8, 4)

        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 64, 3, 1, 1),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.5)
        )

        self.conv21 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, 9, 2, 5),
            #torch.nn.Conv2d(3,96,32,2,0),

            torch.nn.PReLU(),
            torch.nn.MaxPool2d(3, 2)
        )

        self.conv22 = torch.nn.Sequential(
            torch.nn.Conv2d(160, 64, 5, 1, 2),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.5)
        )
        self.conv23 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 5, 1, 2),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.5)
        )
        self.conv24 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 5, 1, 2),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.5)
        )
        self.conv25a = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 5, 1, 2),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.5)
        )
        self.conv25s = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 5, 1, 2),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.5)
        )
        self.deconv_a = torch.nn.ConvTranspose2d(64, 3, 8, 4, 2)
        self.deconv_s = torch.nn.ConvTranspose2d(64, 3, 8, 4, 2)





    def forward(self, x):
        #print(x.size())
        conv1_out = self.conv1(x)
        #print(conv1_out.size())
        conv2_out = self.conv2(conv1_out)
        # print(conv2_out.size())
        conv3_out = self.conv3(conv2_out)
        #print(conv3_out.size())
        conv4_out = self.conv4(conv3_out)
        #print(conv4_out.size())
        conv5_out = self.conv5(conv4_out)
        #print(conv5_out.size())
        upsample_out = self.upsample(conv5_out)
        #print(upsample_out.size())
        conv6_out = self.conv6(upsample_out)
        #print(conv6_out.size())

        conv21_out=self.conv21(x)
        #print(conv21_out.size())
        concat_out=torch.cat([conv21_out,conv6_out],dim=1)
        conv22_out=self.conv22(concat_out)
        conv23_out=self.conv23(conv22_out)
        conv24_out=self.conv24(conv23_out)
        conv25a_out=self.conv25a(conv24_out)
        conv25b_out=self.conv25s(conv24_out)
        #print(conv25a_out.size())

        outa=self.deconv_a(conv25a_out)
        outb=self.deconv_s(conv25b_out)
        #print(outa.size())

        return outa,outb