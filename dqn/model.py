import torch.nn as nn


class DQNModel(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.network = nn.Sequential(
            # convolution with 32 8x8 filters with stride 4
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),

            # convolution with 64 4x4 filters with stride 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),

            # convolution with 64 3x3 filters with stride 1
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),

            nn.Flatten(),

            # fully-connected layer with 512 rectifier units
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.ReLU(),

            # fully-connected layer with a single output for each valid action
            nn.Linear(in_features=512, out_features=n_actions)
        )

    def forward(self, state):
        return self.network(state)
