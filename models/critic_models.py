from models.neural_network import ConvNN, NN

from parameters import Parameters
parameters = Parameters()

critic_num_hidden = parameters.critic_num_hidden
critic_size_hidden = parameters.critic_size_hidden
input_dim = parameters.bot_in_dimension
actor_output_dim = parameters.bot_out_dimension
sidelen = parameters.sidelen
conv_internal_channels = parameters.conv_internal_channels
linear_in_dim = 2 * parameters.number_of_walls + 2
num_conv = parameters.num_conv
convolutional = parameters.convolutional
kernel_size = parameters.conv_kernel_size


class CriticConv(ConvNN):

    def __init__(self):
        """
        Q network class for learning via Sarsa Lambda with convolutional layers.
        """

        self.input_size = input_dim
        self.conv_sidelen = sidelen
        self.conv_in_channels = 4
        self.num_conv = num_conv
        self.conv_internal_channels = conv_internal_channels
        self.linear_in_dim = linear_in_dim
        super().__init__(self.conv_sidelen, self.conv_in_channels,
                         self.conv_internal_channels, self.linear_in_dim + actor_output_dim,
                         critic_size_hidden, self.num_conv, critic_num_hidden, 1, kernel_size)

    def forward(self, x):
        return self.feed_forward(x)


class Critic(NN):

    def __init__(self):
        """
        Q network class for learning via Sarsa Lambda.
        """

        self.input_size = input_dim
        super().__init__(self.input_size, critic_size_hidden, critic_num_hidden, 1)

    def forward(self, x):
        return self.feed_forward(x)
