import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.parallel
#import torch.backends.cudnn as cudnn
#import torch.distributed as dist
#import torch.multiprocessing as mp



## MODEL ARCHITECTURE
class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=(1,1), bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                #subsampling:
                self.shortcut = nn.Sequential()
                if (stride != 1) or (in_channels != out_channels):
                        #if eq. mean downsampling already done by previous block >> why need both clauses?
                        self.shortcut = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=stride, bias=False),
                                nn.BatchNorm2d(out_channels)
                        )

        def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out

class ResNet(nn.Module): #ResNet is a sub-class of nn.Module
        def __init__(self, block, num_blocks, num_classes=10):
                super().__init__()
                self.in_channels = 16
                self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
                self.bn1 = nn.BatchNorm2d(16)
                self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=2)
                self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
                self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
                #maybe self.dropout?
                self.linear = nn.Linear(64, num_classes)

        def _make_layer(self, block, out_channels, num_blocks, stride):
                strides = [stride] + [1]*(num_blocks-1)
                layers = []
                for i in strides:
                        layers.append(block(self.in_channels, out_channels, stride))
                        self.in_channels = out_channels
                return nn.Sequential(*layers)

        def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = F.avg_pool2d(out, 8) # Dims equal to layer to perform global avg pooling
                zero_check = int((out == 0).sum())
                print("BEFORE DROPOUT: " + str(zero_check))
                #print(self.training)
                out = F.dropout2d(out, training=self.training) #default 0.5 ###inplace=True? ###OR MAYBE 3D? ### OR MAYBE FORGOT TRAIN EVAL()?
                zero_check = int((out==0).sum())
                print("AFTER DROPOUT: " + str(zero_check))
                out = out.view(out.size(0), -1) # Reshape to dims [i_rows, ?_columns] from out tensor[i, x, y, ...?]
                out = self.linear(out) ##ALERT: Check out.size() before and after this cmd
                return out


def resnet18():
        return ResNet(BasicBlock, [2, 2, 2])
