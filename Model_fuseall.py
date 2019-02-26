import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# one layer conv net + fc


class Net_2d_conv3_fuse_3fc(nn.Module):

    def __init__(self, dropout = 0.5, levellist=[2, 5, 7], params = [0]):
        super(Net_2d_conv3_fuse_3fc, self).__init__()
        self.in_channels = len(levellist) * len(params)
        self.adjust_dim = False
        if len(levellist) > 1 and len(params) >1:
            self.adjust_dim = True


        self.conv1_uv = nn.Conv2d(in_channels=18, out_channels=64, kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.conv1_bn_uv = nn.BatchNorm2d(64)

        self.conv1_z = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.conv1_bn_z = nn.BatchNorm2d(64)

        self.conv2_uv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.conv2_bn_uv = nn.BatchNorm2d(64)

        self.conv2_z = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.conv2_bn_z = nn.BatchNorm2d(64)


        self.conv3_uv = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.conv3_bn_uv = nn.BatchNorm2d(256)

        self.conv3_z = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.conv3_bn_z = nn.BatchNorm2d(256)

        self.fc1_uv =  nn.Linear(in_features=256*4*4, out_features=576)
        self.fc1_bn_uv = nn.BatchNorm1d(576)
        self.fc1_z =  nn.Linear(in_features=256*4*4, out_features=576)
        self.fc1_bn_z = nn.BatchNorm1d(576)

        self.fc2_uv = nn.Linear(in_features=576, out_features=128)
        self.fc2_bn_uv = nn.BatchNorm1d(128)
        self.fc2_z = nn.Linear(in_features=576, out_features=128)
        self.fc2_bn_z = nn.BatchNorm1d(128)

        self.fc3_uv = nn.Linear(in_features=128, out_features=64)
        self.fc3_bn_uv = nn.BatchNorm1d(64)
        self.fc3_z = nn.Linear(in_features=128, out_features=64)
        self.fc3_bn_z = nn.BatchNorm1d(64)

        self.fc4_uv = nn.Linear(in_features=64, out_features=8)
        self.fc4_bn_uv = nn.BatchNorm1d(8)
        self.fc4_z = nn.Linear(in_features=64, out_features=8)
        self.fc4_bn_z = nn.BatchNorm1d(8)


        self.fc5 = nn.Linear(in_features=8+8+9, out_features=8*3)
        #self.fc5_bn = nn.BatchNorm1d(8)
        self.fc6 = nn.Linear(in_features=8*3, out_features=2*3)
        self.fc7 = nn.Linear(in_features=2*3, out_features=2)
        self.init_weights()

    def init_weights(self):
        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d) and idx==1:
                nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.normal_(m.bias,mean=0, std=1)
            if isinstance(m, nn.Conv2d) and idx!=1:
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
                nn.init.normal_(m.bias,mean=0, std=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and idx==1:
                nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.normal_(m.bias,mean=0, std=1)
            elif isinstance(m, nn.Linear) and idx!=1:
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
                nn.init.normal_(m.bias,mean=0, std=1)

    def forward(self, x, h):
        x1 = x[:,[0,1,3,4,6,7],:,:,:]
        x2 = x[:,[2,5,8],:,:,:]
        if self.adjust_dim:
            x1 = x1.view(x1.shape[0],x1.shape[1]*x1.shape[2], x1.shape[3], x1.shape[4])
            x2 = x2.view(x2.shape[0],x2.shape[1]*x2.shape[2], x2.shape[3], x2.shape[4])


        x1 = F.relu(self.conv1_bn_uv(self.conv1_uv(x1)))
        x1 = F.relu(self.conv2_bn_uv(self.conv2_uv(x1)))
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2, padding=0)
        x1 = F.relu(self.conv3_bn_uv(self.conv3_uv(x1)))
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2, padding=0)
        x1 = x1.view(-1, 256*4*4)

        x2 = F.relu(self.conv1_bn_z(self.conv1_z(x2)))
        x2 = F.relu(self.conv2_bn_z(self.conv2_z(x2)))
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2, padding=0)
        x2 = F.relu(self.conv3_bn_z(self.conv3_z(x2)))
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2, padding=0)
        x2 = x2.view(-1, 256*4*4)

        x1 = F.relu(self.fc1_bn_uv(self.fc1_uv(x1)))
        x1 = F.relu(self.fc2_bn_uv(self.fc2_uv(x1)))
        x1 = F.relu(self.fc3_bn_uv(self.fc3_uv(x1)))
        x1 = F.relu(self.fc4_bn_uv(self.fc4_uv(x1)))

        x2 = F.relu(self.fc1_bn_z(self.fc1_z(x2)))
        x2 = F.relu(self.fc2_bn_z(self.fc2_z(x2)))
        x2 = F.relu(self.fc3_bn_z(self.fc3_z(x2)))
        x2 = F.relu(self.fc4_bn_z(self.fc4_z(x2)))

        x = torch.cat((x1,x2, h), dim=1)

        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = self.fc7(x)
        return x
