import torch.nn as nn


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
