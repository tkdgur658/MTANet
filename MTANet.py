# This code is adapted from the example on https://github.com/Knight825/models-pytorch/blob/master/CrossStagePartial/csp_densenet.py

import torch
from torchvision.models.densenet import _DenseBlock,_DenseLayer,_Transition
import torch.nn as nn
from collections import OrderedDict
import numpy as np

class _Csp_Transition(torch.nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Csp_Transition, self).__init__()
        self.add_module('norm', torch.nn.BatchNorm2d(num_input_features))
        self.add_module('relu', torch.nn.ReLU(inplace=True))
        self.add_module('conv', torch.nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

class _Csp_DenseBlock(torch.nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False,transition = False):
        super(_Csp_DenseBlock,self).__init__()
        self.csp_num_features1 = num_input_features//2
        self.csp_num_features2 = num_input_features - self.csp_num_features1
        trans_in_features = num_layers * growth_rate
        for i in range(num_layers):
            layer = _DenseLayer(
                self.csp_num_features2 + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d'%(i+1),layer)
        self.transition = _Csp_Transition(trans_in_features,trans_in_features//2) if transition else None

    def forward(self,x):
        features = [x[:,self.csp_num_features1:,...]]
        for name,layer in self.named_children():
            if 'denselayer' in name:
                new_feature = layer(features)
                features.append(new_feature)
        dense = torch.cat(features[1:],1)
        if self.transition is not None:
            dense = self.transition(dense)
        return torch.cat([x[:,:self.csp_num_features1,...],dense],1)


class MTANet(torch.nn.Module):
    def __init__(self, in_channels=78, num_classes=1, count_list=[8, 12, 12, 12, 8, 2, 2, 10, 2, 5, 5], growth_rate=32, blocks=(6, 12, 24, 16),
                 num_init_features=64,  transitionBlock = True, transitionDense = True,bn_size=4, drop_rate=0, meta_size = 11,memory_efficient=False):
        super(MTANet,self).__init__()
        self.meta_size = meta_size
        self.count_list = count_list
        if len(self.count_list)!=0:
            self.growth_down_rate = 2 if transitionBlock else 1
            self.count_list = count_list
            self.head_features_list = []

            for count_i,count in enumerate(count_list):

                num_features = num_init_features
                features = torch.nn.Sequential(OrderedDict([
                    ('conv0'+str(count_i+1), torch.nn.Conv2d(count, num_init_features, kernel_size=7, stride=2,
                                        padding=3, bias=False)),
                    ('norm0'+str(count_i+1), torch.nn.BatchNorm2d(num_init_features)),
                    ('relu0'+str(count_i+1), torch.nn.ReLU(inplace=True)),
                    ('pool0'+str(count_i+1), torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]))  
                block = _Csp_DenseBlock(
                    num_layers=blocks[0],
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    memory_efficient=memory_efficient,
                    transition=transitionBlock
                )
                num_features=(num_features+growth_rate*blocks[0])//2
                transition = _Transition(num_input_features=num_features,
                                         num_output_features=num_features//2)
                features.add_module('first_denseblock'+str(count_i+1),block)
                features.add_module('first_transition'+str(count_i+1),transition)
                exec(f'self.head_features_{count_i+1} = features')
                exec(f'self.head_features_list+=[self.head_features_{count_i+1}]')
            self.bottleneck = nn.Sequential(nn.BatchNorm2d(num_features//2*len(count_list) ),nn.ReLU(inplace=True),  nn.Conv2d(num_features//2*len(count_list),num_features//2,1))
            num_features=num_features//2
            self.features =nn.Sequential()
            for i,num_layers in enumerate(blocks):
                if i<=0:
                    continue
                block = _Csp_DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    memory_efficient=memory_efficient,
                    transition=transitionBlock
                )
                self.features.add_module('denseblock%d'%(i+1),block)
                num_features = num_features//2 + num_layers * growth_rate // 2
                if (i != len(blocks)-1) and transitionDense:
                    trans = _Transition(num_input_features=num_features,num_output_features=num_features//2)
                    self.features.add_module('transition%d'%(i+1),trans)
                    num_features = num_features//2

            self.features.add_module('norm5', torch.nn.BatchNorm2d(num_features))

            self.relu = nn.ReLU(inplace=True)
            self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1,1))
            self.classifier = nn.Linear(num_features+self.meta_size, num_classes)

            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.Linear):
                    torch.nn.init.constant_(m.bias, 0)
            self.head_features_list = nn.ModuleList(self.head_features_list)
        else:
            self.classifier = nn.Linear(self.meta_size, num_classes)
    def forward(self, x, meta):
        if len(self.count_list)!=0:
            total_count=0
            for i,  count in enumerate(self.count_list) :
                head_input=x[:,total_count:total_count+count]  
                head_output = self.head_features_list[i](head_input)
                total_count+=count
                if i==0:
                    head_outputs = head_output
                else:
                    head_outputs=torch.cat((head_outputs,head_output),dim=1)
            x = self.bottleneck(head_outputs)
            features = self.features(x)
            out =  self.relu(features)
            out = self.adaptive_avg_pool2d(out)

            out = out.view(out.size(0), -1)
            if self.meta_size!=0:
                out = torch.cat([out,meta],dim = 1)
            out = self.classifier(out)
        else:
            out = self.classifier(meta)
        return out
   
