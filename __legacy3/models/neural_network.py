import torch.nn as nn
from torch import transpose


class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_hidden, out_dim):
        """
        Very generic neural network class. Pretty self-explanatory.
        """

        super(NN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden - 1)]
        for i in range(len(self.hidden_layers)):
            self.add_module("hidden_layer_" + str(i), self.hidden_layers[i])
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def feed_forward(self, x):
        y = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            y = self.relu(layer(y))
        output = self.output_layer(y)
        return output

    def pull(self, model):
        self.load_state_dict(model.state_dict())

    def zero_all_parameters(self):
        for param in self.parameters():
            param.data.zero_()


class ConvNN(nn.Module):
    def __init__(self, conv_sidelen, conv_in_channels, conv_internal_channels,
                 linear_in_dim, hidden_dim, num_conv, num_hidden, out_dim,
                 kernel_size, stride=1, padding=0, maxpool_kernel=(1, 1)):
        """
        Very generic neural network class. Pretty self-explanatory.
        """

        super(ConvNN, self).__init__()
        self.conv_sidelen = conv_sidelen
        self.conv_in_channels = conv_in_channels
        self.conv_internal_channels = conv_internal_channels
        self.num_conv = num_conv
        self.linear_in_dim = linear_in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel = maxpool_kernel
        self.conv_out_dim = self.conv_sidelen + 2 * (1 + self.num_conv) * padding \
            - (1 + self.num_conv) * (self.kernel_size[0] - 1)
        self.conv_out_size = self.conv_sidelen ** 2 * self.conv_internal_channels
        self.input_layer_walls = nn.Linear(linear_in_dim, hidden_dim)
        self.input_layer_conv = nn.Conv2d(conv_in_channels, conv_internal_channels,
                                          kernel_size=self.kernel_size, stride=self.stride,
                                          padding=self.padding)
        self.hidden_layers_conv = [nn.Conv2d(conv_internal_channels, conv_internal_channels,
                                             kernel_size=self.kernel_size, stride=self.stride,
                                             padding=self.padding) for _ in range(self.num_conv)]
        self.pool = nn.MaxPool2d(kernel_size=self.maxpool_kernel)
        self.flat = nn.Flatten()
        self.hidden_conversion_layer = nn.Linear(self.conv_out_size, hidden_dim)
        self.hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden - 1)]
        for i in range(len(self.hidden_layers)):
            self.add_module("hidden_layer_" + str(i), self.hidden_layers[i])
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def feed_forward(self, x):

        # convolutions
        conv_part = x[:-self.linear_in_dim].reshape((self.conv_sidelen, self.conv_sidelen, self.conv_in_channels))
        conv_part = transpose(conv_part, 0, 2)
        conv_part = transpose(conv_part, 1, 2)
        conv_part = conv_part.reshape((1, self.conv_in_channels, self.conv_sidelen, self.conv_sidelen))

        y_conv = self.relu(self.input_layer_conv(conv_part))
        for layer in self.hidden_layers_conv:
            y_conv = self.relu(layer(y_conv))
        y_conv = self.flat(self.pool(y_conv))
        y_conv = self.relu(self.hidden_conversion_layer(y_conv))

        # get linear part up to size
        linear_part = x[-self.linear_in_dim:]
        y_linear = self.relu(self.input_layer_walls(linear_part))

        # sum the parts
        y = y_linear + y_conv

        # handle final feedforward layers
        for layer in self.hidden_layers:
            y = self.relu(layer(y))
        output = self.output_layer(y)
        return output

    def pull(self, model):
        self.load_state_dict(model.state_dict())
