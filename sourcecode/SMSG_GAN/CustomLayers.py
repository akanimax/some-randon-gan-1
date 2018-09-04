""" Module containing custom layers """
import torch as th
import copy


# ==========================================================
# Extra Layers (for experimentation purposes)
# ==========================================================

class SelfAttention(th.nn.Module):
    """
    Layer implements the self-attention module
    which is the main logic behind this architecture.
    Mechanism described in the paper ->
    Self Attention GAN: refer /literature/Zhang_et_al_2018_SAGAN.pdf
    args:
        channels: number of channels in the image tensor
        activation: activation function to be applied (default: lrelu(0.2))
        squeeze_factor: squeeze factor for query and keys (default: 8)
        bias: whether to apply bias or not (default: True)
    """
    from torch.nn import LeakyReLU

    def __init__(self, channels, activation=LeakyReLU(0.2), squeeze_factor=8, bias=True):
        """ constructor for the layer """

        from torch.nn import Conv2d, Parameter, Softmax

        # base constructor call
        super().__init__()

        # state of the layer
        self.activation = activation
        self.gamma = Parameter(th.zeros(1))

        # Modules required for computations
        self.query_conv = Conv2d(  # query convolution
            in_channels=channels,
            out_channels=channels // squeeze_factor,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=bias
        )

        self.key_conv = Conv2d(  # key convolution
            in_channels=channels,
            out_channels=channels // squeeze_factor,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=bias
        )

        self.value_conv = Conv2d(  # value convolution
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=bias
        )

        # softmax module for applying attention
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        forward computations of the layer
        :param x: input feature maps (B x C x H x W)
        :return:
            out: self attention value + input feature (B x O x H x W)
            attention: attention map (B x H x W x H x W)
        """

        # extract the shape of the input tensor
        m_batchsize, c, height, width = x.size()

        # create the query projection
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)  # B x (N) x C

        # create the key projection
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height)  # B x C x (N)

        # calculate the attention maps
        energy = th.bmm(proj_query, proj_key)  # energy
        attention = self.softmax(energy)  # attention B x (N) x (N)

        # create the value projection
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X (N)

        # calculate the output
        out = th.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, c, height, width)

        attention = attention.view(m_batchsize, height, width, height, width)

        if self.activation is not None:
            out = self.activation(out)

        # apply the residual connection
        out = (self.gamma * out) + x
        return out, attention


# ==========================================================
# Layers required for Building The generator and
# discriminator
# ==========================================================

class GenInitialBlock(th.nn.Module):
    """ Module implementing the initial block of the Generator
        Takes in whatever latent size and generates output volume
        of size 4 x 4
    """

    def __init__(self, in_channels):
        """
        constructor for the inner class
        :param in_channels: number of input channels to the block
        """
        from torch.nn import LeakyReLU
        from torch.nn import Conv2d, ConvTranspose2d
        super().__init__()

        self.conv_1 = ConvTranspose2d(in_channels, in_channels, (4, 4), bias=True)
        self.conv_2 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)
        self.self_attention = SelfAttention(in_channels, squeeze_factor=8)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input to the module
        :return: y => output
        """
        # convert the tensor shape:
        y = x.view(*x.shape, 1, 1)  # add two dummy dimensions for
        # convolution operation

        # perform the forward computations:
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))
        y, attention = self.self_attention(y)

        return y, attention


class GenGeneralConvBlock(th.nn.Module):
    """ Module implementing a general convolutional block """

    def __init__(self, in_channels, out_channels):
        """
        constructor for the class
        :param in_channels: number of input channels to the block
        :param out_channels: number of output channels required
        """
        from torch.nn import LeakyReLU

        super().__init__()

        from torch.nn import Conv2d
        self.conv_1 = Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=True)
        self.conv_2 = Conv2d(out_channels, out_channels, (3, 3), padding=1, bias=True)
        self.self_attention = SelfAttention(out_channels, squeeze_factor=8)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import interpolate

        y = interpolate(x, scale_factor=2)
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))
        y, attention = self.self_attention(y)

        return y, attention


class MinibatchStdDev(th.nn.Module):
    def __init__(self, averaging='all'):
        """
        constructor for the class
        :param averaging: the averaging mode used for calculating the MinibatchStdDev
        """
        super().__init__()

        # lower case the passed parameter
        self.averaging = averaging.lower()

        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in \
                   ['all', 'flat', 'spatial', 'none', 'gpool'], \
                   'Invalid averaging mode %s' % self.averaging

        # calculate the std_dev in such a way that it doesn't result in 0
        # otherwise 0 norm operation's gradient is nan
        self.adjusted_std = lambda x, **kwargs: th.sqrt(
            th.mean((x - th.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):
        """
        forward pass of the Layer
        :param x: input
        :return: y => output
        """
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)

        # compute the std's over the minibatch
        vals = self.adjusted_std(x, dim=0, keepdim=True)

        # perform averaging
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = th.mean(vals, dim=1, keepdim=True)

        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = th.mean(th.mean(vals, 2, keepdim=True), 3, keepdim=True)

        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]

        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = th.mean(th.mean(th.mean(x, 2, keepdim=True),
                                       3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = th.FloatTensor([self.adjusted_std(x)])

        else:  # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1] /
                             self.n, self.shape[2], self.shape[3])
            vals = th.mean(vals, 0, keepdim=True).view(1, self.n, 1, 1)

        # spatial replication of the computed statistic
        vals = vals.expand(*target_shape)

        # concatenate the constant feature map to the input
        y = th.cat([x, vals], 1)

        # return the computed value
        return y


class DisFinalBlock(th.nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels):
        """
        constructor of the class
        :param in_channels: number of input channels
        """
        from torch.nn import LeakyReLU
        from torch.nn import Conv2d

        super().__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()

        # modules required:
        self.self_attention = SelfAttention(in_channels, squeeze_factor=8)
        self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
        self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)

        # final conv layer emulates a fully connected layer
        self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y, _ = self.self_attention(y)
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y.view(-1)


class DisGeneralConvBlock(th.nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        """
        from torch.nn import AvgPool2d, LeakyReLU
        from torch.nn import Conv2d

        super().__init__()

        # convolutional modules
        self.self_attention = SelfAttention(in_channels, squeeze_factor=8)
        self.conv_1 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)
        self.conv_2 = Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=True)
        self.downSampler = AvgPool2d(2)  # downsampler

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y, _ = self.self_attention(x)
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)

        return y
