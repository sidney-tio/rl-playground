import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMLPNetwork(nn.Module):
    """"CNN and MLP network to process OvercookedAI world_state.
    Adapted from https://github.com/HumanCompatibleAI/human_aware_rl/blob/master/human_aware_rl/baselines_utils.py"""

    def __init__(self, input_size, output_size, params):
        super(ConvMLPNetwork, self).__init__()
        num_hidden_layers = params["NUM_HIDDEN_LAYERS"]
        size_hidden_layers = params["SIZE_HIDDEN_LAYERS"]
        num_filters = params["NUM_FILTERS"]
        num_convs = params["NUM_CONV_LAYERS"]
        shape = params["OBS_STATE_SHAPE"]

        # Need to double check the padding and calculations
        self.conv_initial = nn.Conv2d(input_size, num_filters, 5,
                                      padding=self.calculate_padding(5, 'same'))
        self.conv_layers = nn.ModuleList([nn.Conv2d(
            num_filters, num_filters, kernel_size=3, padding=self.calculate_padding(5, 'same')) for i in range(0, num_convs - 2)])
        self.conv_final = nn.Conv2d(num_filters, num_filters, kernel_size=3,
                                    padding=self.calculate_padding(3, 'valid'))
        self.linear_initial = nn.Linear(shape[0] * shape[1] * num_filters, size_hidden_layers)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(size_hidden_layers, size_hidden_layers) for i in range(num_hidden_layers)])
        self.linear_final = nn.Linear(size_hidden_layers, output_size)

    def forward(self, x):
        x = self.conv_initial(x)
        for convs in self.conv_layers:
            x = F.leaky_relu(convs(x))
        x = self.conv_final(x)
        x = x.view(x.size()[0], -1)
        x = self.linear_initial(x)
        for linear in self.linear_layers:
            x = F.leaky_relu(linear(x))
        x = F.softmax(self.linear_final(x))
        return x

    def calculate_padding(self, kernel, pad_type):
        if pad_type == 'valid':
            return 0
        if pad_type == 'same':
            return int((kernel - 1) / 2)


class MLPNetwork(nn.Module):

    def __init__(self, params, obs_space, action_space):
        super(MLPNetwork, self).__init__()

        num_layers = params['num_layers']
        layer_size = params['layer_size']
        initial_layer_size = layer_size[0]
        output_size = action_space
        assert num_layers == len(layer_size)

        self.input_layer = nn.Linear(obs_space[0], initial_layer_size)

        self.linear_layers = nn.ModuleList(
            [nn.Linear(layer_size[i], layer_size[i+1]) for i in range(0, num_layers - 1)])

        if layer_size:
            self.output_actor = nn.Linear(layer_size[-1], output_size)
            self.output_critic = nn.Linear(layer_size[-1], 1)
        else:
            self.output_actor = nn.Linear(initial_layer_size, output_size)
            self.output_critic = nn.Linear(initial_layer_size, 1)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for layers in self.linear_layers:
            x = F.relu(layers(x))

        prob = F.softmax(self.output_actor(x), dim=-1)
        value = self.output_critic(x)
        return prob, value

    def act(self, state, action=None):
        prob, value = self.forward(state)
        dist = torch.distributions.Categorical(prob.squeeze())
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value
