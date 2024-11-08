import torch
import torch.nn as nn
from timm.models import vgg16
import torch.nn.functional as F
import antialiased_cnns as aac
from antialiased_cnns import BlurPool
import ipdb
import numpy as np


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class LayerActivation:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()

""" def encoder(input_layer):
    VGG_MEAN = [103.939, 116.779, 123.68]

    # Convert RGB to BGR
    red, green, blue = tf.split(input_layer.outputs, 3, 3)
    bgr = tf.concat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], axis=3)

    network = tl.layers.InputLayer(bgr, name='encoder/input_layer_bgr')

    # Convolutional layers size 1
    network = conv_layer(network, [3, 64], 'encoder/h1/conv_1')
    beforepool1 = conv_layer(network, [64, 64], 'encoder/h1/conv_2')
    network = pool_layer(beforepool1, 'encoder/h1/pool')

    # Convolutional layers size 2
    network = conv_layer(network, [64, 128], 'encoder/h2/conv_1')
    beforepool2 = conv_layer(network, [128, 128], 'encoder/h2/conv_2')
    network = pool_layer(beforepool2, 'encoder/h2/pool')

    # Convolutional layers size 3
    network = conv_layer(network, [128, 256], 'encoder/h3/conv_1')
    network = conv_layer(network, [256, 256], 'encoder/h3/conv_2')
    beforepool3 = conv_layer(network, [256, 256], 'encoder/h3/conv_3')
    network = pool_layer(beforepool3, 'encoder/h3/pool')

    # Convolutional layers size 4
    network = conv_layer(network, [256, 512], 'encoder/h4/conv_1')
    network = conv_layer(network, [512, 512], 'encoder/h4/conv_2')
    beforepool4 = conv_layer(network, [512, 512], 'encoder/h4/conv_3')
    network = pool_layer(beforepool4, 'encoder/h4/pool')

    # Convolutional layers size 5
    network = conv_layer(network, [512, 512], 'encoder/h5/conv_1')
    network = conv_layer(network, [512, 512], 'encoder/h5/conv_2')
    beforepool5 = conv_layer(network, [512, 512], 'encoder/h5/conv_3')
    network = pool_layer(beforepool5, 'encoder/h5/pool')

    return network, (input_layer, beforepool1, beforepool2, beforepool3, beforepool4, beforepool5) 
    
    
# Convolutional layer
def conv_layer(input_layer, sz, str):
    network = tl.layers.Conv2dLayer(input_layer,
                                    act=tf.nn.relu,
                                    shape=[3, 3, sz[0], sz[1]],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name=str)

    return network


# Max-pooling layer
def pool_layer(input_layer, str):
    network = tl.layers.PoolLayer(input_layer,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name=str)

    return network
    
    """


class Encoder(nn.Module):
    def __init__(self,device,in_channels=5):
        super(Encoder, self).__init__()
        self.vgg = vgg16(pretrained=True)
        self.device=device
        # for param in self.vgg.parameters():
        #     param.requires_grad = False
        self.skip1 = LayerActivation(self.vgg.features, 3)
        self.skip2 = LayerActivation(self.vgg.features, 8)
        self.skip3 = LayerActivation(self.vgg.features, 15)
        self.skip4 = LayerActivation(self.vgg.features, 22)
        self.skip5 = LayerActivation(self.vgg.features, 29)
        self.input_layer = Conv2d(in_channels, 3, kernel_size=3,padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        self.vgg(x)

        return x, self.skip1.features, self.skip2.features \
            , self.skip3.features, self.skip4.features, self.skip5.features,


class Encoder_aa(nn.Module):
    def __init__(self,device,in_channels=5):
        super(Encoder_aa, self).__init__()
        self.vgg = aac.vgg16(pretrained=True)
        self.device=device
        # for param in self.vgg.parameters():
        #     param.requires_grad = False
        self.skip1 = LayerActivation(self.vgg.features, 3)
        self.skip2 = LayerActivation(self.vgg.features, 9)
        self.skip3 = LayerActivation(self.vgg.features, 17)
        self.skip4 = LayerActivation(self.vgg.features, 25)
        self.skip5 = LayerActivation(self.vgg.features, 33)
        self.input_layer = Conv2d(in_channels, 3, kernel_size=3,padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        self.vgg(x)

        return x, self.skip1.features, self.skip2.features \
            , self.skip3.features, self.skip4.features, self.skip5.features,



class Decoder(nn.Module):
    def __init__(self, device,out_channels=4):
        super(Decoder, self).__init__()
        self.device=device
        self.latent_representation = nn.Sequential(
            Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            Conv2d(512, 512, kernel_size=3, padding=1)
        )
        self.convTranspose_5 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.conv1x1_5 = Conv2d(1024, 512, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.convTranspose_4 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.conv1x1_4 = Conv2d(1024, 512, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.convTranspose_3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv1x1_3 = Conv2d(512, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.convTranspose_2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv1x1_2 = Conv2d(256, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.convTranspose_1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv1x1_1 = Conv2d(128, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv1x1_64_4 = nn.Conv2d(64, 4, kernel_size=1)
        self.conv1x1_7_4 = nn.Conv2d(7, out_channels, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(4)

    def upsample(self,x, convT,bn, skip, conv1x1,):
        x = convT(x)
        x = bn(x)
        x = F.leaky_relu(x, 0.2)

        #skip = torch.log(skip ** 2 + 1.0/255.0)
        x = torch.cat([x, skip], dim=1)
        x = conv1x1(x)
        return x

    def upsample_last(self,x, conv1x1_64_4,bn, skip, conv1x1_8_4):
        x = conv1x1_64_4(x)
        x = bn(x)
        x = F.leaky_relu(x, 0.2)

        #skip = torch.log(skip ** 2 + 1.0 / 255.0)
        x = torch.cat([x, skip], dim=1)
        x = conv1x1_8_4(x)
        return x


    def forward(self, skip0, skip1, skip2, skip3, skip4, skip5):
        x = self.latent_representation(skip5)
        x = self.upsample(x, self.convTranspose_5,self.bn5, skip5, self.conv1x1_5)
        x = self.upsample(x, self.convTranspose_4,self.bn4, skip4, self.conv1x1_4)
        x = self.upsample(x, self.convTranspose_3,self.bn3, skip3, self.conv1x1_3)
        x = self.upsample(x, self.convTranspose_2,self.bn2, skip2, self.conv1x1_2)
        x = self.upsample(x, self.convTranspose_1,self.bn1, skip1, self.conv1x1_1)
        x = self.upsample_last(x, self.conv1x1_64_4,self.bn0, skip0, self.conv1x1_7_4)
        return x
    
class BlurPool_Transpose(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool_Transpose, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
        
def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer
    

class Decoder_aa(nn.Module):
    def __init__(self, device,out_channels=4):
        super(Decoder_aa, self).__init__()
        self.device=device
        self.latent_representation = nn.Sequential(
            Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            Conv2d(512, 512, kernel_size=3, padding=1)
        )
        self.convTranspose_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            BlurPool_Transpose(512,stride=2)
        )
        self.conv1x1_5 = Conv2d(1024, 512, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(512)


        self.convTranspose_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            BlurPool_Transpose(512,stride=2)
        )
        self.conv1x1_4 = Conv2d(1024, 512, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(512)


        self.convTranspose_3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
            BlurPool_Transpose(256,stride=2)
        )
        self.conv1x1_3 = Conv2d(512, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(256)


        self.convTranspose_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            BlurPool_Transpose(128,stride=2)
        )
        self.conv1x1_2 = Conv2d(256, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(128)


        self.convTranspose_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            BlurPool_Transpose(64,stride=2)
        )
        self.conv1x1_1 = Conv2d(128, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)


        self.conv1x1_64_4 = nn.Conv2d(64, 4, kernel_size=1)
        self.conv1x1_7_4 = nn.Conv2d(7, out_channels, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(4)

    def upsample(self,x, convT,bn, skip, conv1x1):
        x = convT(x)
        x = bn(x)
        x = F.leaky_relu(x, 0.2)

        #skip = torch.log(skip ** 2 + 1.0/255.0)
        x = torch.cat([x, skip], dim=1)
        x = conv1x1(x)
        return x

    def upsample_last(self,x, conv1x1_64_4,bn, skip, conv1x1_8_4):
        x = conv1x1_64_4(x)
        x = bn(x)
        x = F.leaky_relu(x, 0.2)

        #skip = torch.log(skip ** 2 + 1.0 / 255.0)
        x = torch.cat([x, skip], dim=1)
        x = conv1x1_8_4(x)
        return x


    def forward(self, skip0, skip1, skip2, skip3, skip4, skip5):
        x = self.latent_representation(skip5)
        x = self.upsample(x, self.convTranspose_5,self.bn5, skip5, self.conv1x1_5)
        x = self.upsample(x, self.convTranspose_4,self.bn4, skip4, self.conv1x1_4)
        x = self.upsample(x, self.convTranspose_3,self.bn3, skip3, self.conv1x1_3)
        x = self.upsample(x, self.convTranspose_2,self.bn2, skip2, self.conv1x1_2)
        x = self.upsample(x, self.convTranspose_1,self.bn1, skip1, self.conv1x1_1)
        x = self.upsample_last(x, self.conv1x1_64_4,self.bn0, skip0, self.conv1x1_7_4)
        return x

class Bottleneck(nn.Module):
    def __init__(self, device):
        super(Bottleneck, self).__init__()
        self.device = device
        self.layer = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(512)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        # bottleneck
        x = self.layer(x)
        x = self.batch_norm(x)
        x = self.act(x)
        return x

class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.encoder = Encoder(device)
        self.decoder = Decoder(device)

    def forward(self, x):
        x = x.float()
        # encoder
        skip0, skip1, skip2, skip3, skip4, skip5 = self.encoder(x)

        #decoder
        x = self.decoder(skip0, skip1, skip2, skip3, skip4, skip5)
        return x
    
