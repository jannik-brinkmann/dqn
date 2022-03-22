import torch.nn as nn

class DQNModelCartPole(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class DQNModel(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.convolution = nn.Sequential(
            # convolution with 32 8x8 filters with stride 4 and applies rectifier non-linearity
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),

            # convolution with 64 4x4 filters with stride 2 and applies rectifier non-linearity
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),

            # convolution with 64 3x3 filters with stride 1 and applies rectifier non-linearity
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            # fully-connected layer with 512 rectifier units
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.ReLU(),

            # output layer with a single output for each valid action
            nn.Linear(in_features=512, out_features=n_actions)
        )

    def forward(self, state):
        features = self.convolution(state)
        q_values = self.linear(features.view(features.size(0), -1))
        return q_values
