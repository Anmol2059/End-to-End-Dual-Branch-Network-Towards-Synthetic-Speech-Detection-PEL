import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
import os
import random
import numpy as np

class ChannelAttention(nn.Module):           # Channel Attention Module
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):           # Spatial Attention Module
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()

        #self.output_size = output_size
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),requires_grad=True)

        self.mean_only = mean_only

        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        if inputs.size(0)==1:
            attentions = F.softmax(torch.tanh(weights),dim=1)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            attentions = F.softmax(torch.tanh(weights.squeeze()),dim=1)
            weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

        if self.mean_only:
            return weighted.sum(1)
        else:
            noise = 1e-5*torch.randn(weighted.size())

            if inputs.is_cuda:
                noise = noise.to(inputs.device)
            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)

            representations = torch.cat((avg_repr,std_repr),1)

            return representations


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.channel = ChannelAttention(self.expansion * planes)  # Channel Attention Module
        self.spatial = SpatialAttention()  # Spatial Attention Module


        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        CBAM_Cout = self.channel(out)
        out = out * CBAM_Cout
        CBAM_Sout = self.spatial(out)
        out = out * CBAM_Sout
        out += shortcut
        return out



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

RESNET_CONFIGS = {'18': [[2, 2, 2, 2], PreActBlock]}

def setup_seed(random_seed, cudnn_deterministic=True):
    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = cudnn_deterministic


class SelfEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(SelfEnhancementModule, self).__init__()
        self.median_filter = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.median_filter.weight.data.fill_(1.0 / 9.0)  # Initialize as an averaging filter
        self.noise_enhancement = nn.Sequential(
            nn.Sigmoid(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Noise Enhancement
        noise = x - self.median_filter(x)
        noise = self.noise_enhancement(noise)
        enhanced_features = x + noise

        # Channel Attention
        ca = self.channel_attention(enhanced_features)
        return enhanced_features * ca

class MutualEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(MutualEnhancementModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=1)
        attention = self.attention(combined)
        return x1 * attention, x2 * attention

class ResNet(nn.Module):
    def __init__(self, num_nodes, enc_dim, resnet_type='18', nclasses=2):
        self.in_planes = 16
        super(ResNet, self).__init__()

        layers, block = RESNET_CONFIGS[resnet_type]

        self._norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.LeakyReLU()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv5 = nn.Conv2d(512 * block.expansion, 256, kernel_size=(num_nodes, 3), stride=(1, 1), padding=(0, 1),
                               bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 2, enc_dim)
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

        self.self_enhancement = SelfEnhancementModule(256)
        self.mutual_enhancement = MutualEnhancementModule(256)
        self.attention = SelfAttention(256)

        self.initialize_params()

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(self.bn1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        intermediate_output = self.layer4(x)
        intermediate_output = self.conv5(intermediate_output)
        intermediate_output = self.activation(self.bn5(intermediate_output)).squeeze(2)

        if intermediate_output.dim() == 3:
            intermediate_output = intermediate_output.unsqueeze(-1)  # Add the missing dimension

        # Check the shape before passing it to the SelfEnhancementModule
        print("Shape before SelfEnhancementModule:", intermediate_output.shape)

        # Apply Self Enhancement Module
        self_enhanced = self.self_enhancement(intermediate_output)

        # Check the shape after SelfEnhancementModule
        print("Shape after SelfEnhancementModule:", self_enhanced.shape)

        # Pass the enhanced features to the Mutual Enhancement Module
        mutual_enhanced_lfcc, mutual_enhanced_cqt = self.mutual_enhancement(self_enhanced, self_enhanced)

        # Remove the last dimension from the mutual enhanced LFCC
        mutual_enhanced_lfcc = mutual_enhanced_lfcc.squeeze(-1)

        # Apply Self Attention
        stats = self.attention(mutual_enhanced_lfcc.permute(0, 2, 1).contiguous())
        feat = self.fc(stats)
        mu = self.fc_mu(feat)
        print("my shape:", mu.shape)
        print("feature shape", feat.shape)




        return feat, mu , intermediate_output



class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# class TypeClassifier(nn.Module):
#     def __init__(self, enc_dim, nclasses, lambda_=0.05, ADV=True):
#         super(TypeClassifier, self).__init__()
#         self.adv = ADV
#         if self.adv:
#             self.grl = GradientReversal(lambda_)
#         self.classifier = nn.Sequential(nn.Linear(enc_dim, enc_dim // 2),
#                                         nn.Dropout(0.3),
#                                         nn.ReLU(),
#                                         nn.Linear(enc_dim // 2, nclasses),
#                                         nn.ReLU())

#     def initialize_params(self):
#         for layer in self.modules():
#             if isinstance(layer, torch.nn.Linear):
#                 init.kaiming_uniform_(layer.weight)

#     def forward(self, x):
#         # Flatten 
#         if x.dim() > 2:
#             x = x.view(x.size(0), -1)

#         if self.adv:
#             x = self.grl(x)
        
#         return self.classifier(x)

class TypeClassifier(nn.Module):
    def __init__(self, enc_dim, nclasses, lambda_=0.05, ADV=True):
        super(TypeClassifier, self).__init__()
        self.adv = ADV
        if self.adv:
            self.grl = GradientReversal(lambda_)
        self.classifier = nn.Sequential(nn.Linear(enc_dim, enc_dim // 2),
                                        nn.Dropout(0.3),
                                        nn.ReLU(),
                                        nn.Linear(enc_dim // 2, nclasses),
                                        nn.ReLU())

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        if self.adv:
            x = self.grl(x)
        return self.classifier(x)



