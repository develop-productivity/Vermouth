
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys 
sys.path.append('/data/sydong/diffusion/diffusion_features')
from src.model.modules import SimpleFuse, AttentionPool2d

class Linear_prob(nn.Module):
    def __init__(self, in_channels=1280,class_num=1000):
        super().__init__()
        self.adap_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_channels, class_num)
        # for m in self.linear.parameters():
        #     m.data.fill_(0.1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.adap_pool(x)
        x = x.view(batch_size, -1)
        pre = self.linear(x)
        # pre = torch.sigmoid(self.linear(x))
        return pre
    

class Adapter(nn.Module):
    def __init__(self, in_feature=1280, out_channels=768) -> None:
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_channels
        reduction=4
        self.map =  nn.Sequential(
            nn.Conv2d(self.in_feature, self.in_feature // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.in_feature // reduction, self.in_feature, kernel_size=1, stride=1, padding=0, bias=False),
        )
    def forward(self, inputs):
        if isinstance(inputs, list):
            inputs = inputs[0]
        out = self.map(inputs)
        return out
    

class FuseClassifierPre(SimpleFuse):
    def __init__(self, in_dims, out_res=4, clip_feature_dim=768, arch='tiny', do_fuse=True):
        """
        :param in_dims: list of int, input dimensions   low res to high res
        :param out_dims: list of int, output dimensions  low res to high res
        :param out_res: int, output resolution
        """
        if do_fuse:
            fuse_method = 'high2low'
        else:
            fuse_method = 'no_fuse'
        super(FuseClassifierPre, self).__init__(in_dims, arch=arch, fuse_method=fuse_method)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((out_res, out_res))
        self.clip_feature=768
        self.attn_pool = AttentionPool2d(4, self.out_dims[0], num_heads=8, output_dim=self.clip_feature)
        self.clip_feature_dim = clip_feature_dim
        self.do_fuse=do_fuse
        self.out_res=out_res

    def forward(self, out_list):
        low_res, mid_res, high_res, highest_res = out_list
        if self.do_fuse:
            highest_res = self.highest_res_conv(highest_res) 
            highest_res = self.avg_pool(highest_res)

            high_res = torch.cat([high_res, highest_res], dim=1)
            high_res = self.high_res_conv(high_res)
            high_res = self.avg_pool(high_res)

            mid_res = torch.cat([mid_res, high_res], dim=1)
            mid_res = self.mid_res_conv(mid_res)
            mid_res = self.avg_pool(mid_res)

            low_res = torch.cat([low_res, mid_res], dim=1)
            low_res = self.low_res_conv(low_res)
        else:
            low_res = self.low_res_conv(low_res)
        if low_res.shape[-1] != self.out_res:
            low_res = self.adaptive_pool(low_res)

        return self.attn_pool(low_res)
    
class FuseClassifier(SimpleFuse):
    def __init__(self, in_dims, out_res=4, clip_feature_dim=768, arch='tiny', do_fuse=True):
        """
        :param in_dims: list of int, input dimensions   low res to high res
        :param out_dims: list of int, output dimensions  low res to high res
        :param out_res: int, output resolution
        """
        if do_fuse:
            fuse_method = 'high2low'
        else:
            fuse_method = 'no_fuse'
        super(FuseClassifier, self).__init__(in_dims, arch=arch, fuse_method=fuse_method)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((out_res, out_res))
        self.clip_feature_dim = clip_feature_dim
        self.do_fuse=do_fuse
        self.out_res=out_res

    def forward(self, out_list):
        low_res, mid_res, high_res, highest_res = out_list
        if self.do_fuse:
            highest_res = self.highest_res_conv(highest_res) 
            highest_res = self.avg_pool(highest_res)

            high_res = torch.cat([high_res, highest_res], dim=1)
            high_res = self.high_res_conv(high_res)
            high_res = self.avg_pool(high_res)

            mid_res = torch.cat([mid_res, high_res], dim=1)
            mid_res = self.mid_res_conv(mid_res)
            mid_res = self.avg_pool(mid_res)

            low_res = torch.cat([low_res, mid_res], dim=1)
            low_res = self.low_res_conv(low_res)
        else:
            low_res = self.low_res_conv(low_res)
        if low_res.shape[-1] != self.out_res:
            low_res = self.adaptive_pool(low_res)

        return low_res

class ccbn(nn.Module):
    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
               cross_replica=False, mybn=False, norm_style='bn',):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # Norm style?
        self.norm_style = norm_style
        
        if self.cross_replica:
            self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
        elif self.mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        elif self.norm_style in ['bn', 'in']:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var',  torch.ones(output_size)) 
    
    
    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        # gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        # bias = self.bias(y).view(y.size(0), -1, 1, 1)
        # # If using my batchnorm
        # if self.mybn or self.cross_replica:
        #     return self.bn(x, gain=gain, bias=bias)
        # # else:
        # else:
        #     if self.norm_style == 'bn':
        #         out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
        #                         self.training, 0.1, self.eps)
        #     elif self.norm_style == 'in':
        #         out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
        #                         self.training, 0.1, self.eps)
        #     elif self.norm_style == 'gn':
        #         out = groupnorm(x, self.normstyle)
        #     elif self.norm_style == 'nonorm':
        #         out = x
        #     return out * gain + bias
        return x

    def extra_repr(self):
        s = 'out: {output_size}, in: {input_size},'
        s +=' cross_replica={cross_replica}'
        return s.format(**self.__dict__)


class SegBlock(nn.Module):
    def __init__(self, in_channels, out_channels, con_channels,
                which_conv=nn.Conv2d, which_linear=None, activation=None, 
                upsample=None):
        super(SegBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_linear = which_conv, which_linear
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                            kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = ccbn(in_channels, con_channels, self.which_linear, eps=1e-4, norm_style='bn')
        self.bn2 = ccbn(out_channels, con_channels, self.which_linear, eps=1e-4, norm_style='bn')
        # upsample layers
        self.upsample = upsample

    def forward(self, x, y):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:       
            x = self.conv_sc(x)
        return h + x



def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
    # adapter = Adapter(in_channels=1280)
    # linear = Linear_prob(1280, 500)
    in_dims=[1280, 1357, 1357, 960]
    expert_biases = [512, 256, 128]
    in_dims[1:] = [in_dim + expert_bias for in_dim, expert_bias in zip(in_dims[1:], expert_biases)]  # for resnet expert
    fuser_classifier = FuseClassifier(in_dims=in_dims, arch='tiny', out_res=4)
    # res18 = torchvision.models.resnet18()

    # print('===============adpter================')
    # print('adapter paramters: ', get_parameter_number(adapter))
    # print('===============linear================')
    # print('linear paramters: ', get_parameter_number(linear))


    # Mix_conv = SegBlock(1280, 1280, 768, which_conv=nn.Conv2d, which_linear=nn.Linear, activation=nn.ReLU(inplace=True), upsample=nn.Upsample(scale_factor=2, mode='bilinear'))
    print('===============fuser_classifier================')
    print('fuse paramters: ', get_parameter_number(fuser_classifier))
    # print('===============resnet18================')
    # print('res18 paramters: ', get_parameter_number(res18))
