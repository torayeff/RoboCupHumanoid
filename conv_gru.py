import torch
import torch.nn as nn


class ConvGruCell(nn.Module):
    """Implementation of Convolution GRU cell as described in https://arxiv.org/pdf/1511.06432.pdf
    """

    def __init__(self, input_channels, hidden_dim, device=None, kernel_size=3):
        super(ConvGruCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim  # TODO: Check if we let this the same as the input dim.
        self.kernel_size = kernel_size
        self.padding = kernel_size // 3
        self.device = device

        self.conv_zt = nn.Conv2d(self.input_channels + self.hidden_dim,
                                 self.hidden_dim, padding=self.padding, kernel_size=self.kernel_size)
        self.conv_rt = nn.Conv2d(self.input_channels + self.hidden_dim,
                                 self.hidden_dim, padding=self.padding, kernel_size=self.kernel_size)
        self.conv_h_hat = nn.Conv2d(self.input_channels + self.hidden_dim,
                                    self.hidden_dim, padding=self.padding, kernel_size=self.kernel_size)

    def forward(self, input, h_t=None):
        """Forward pass implementation of the ConvGruCell

        Args
            input: input tensor of shape (batch/seq_len, channel, height, width)
            h_t: previous hidden vector of shape (1, hidden_dim, height, width)
        """

        if h_t is None:
            h_t = torch.zeros(1, self.hidden_dim, input.size(2), input.size(3), device=self.device)

        self.test_sizes(input, h_t)

        seq_len = input.size(0)

        for i in range(seq_len):
            x_t = input[i].unsqueeze(0)
            stacked_x_hidden = torch.cat([x_t, h_t], dim=1)

            z_t = nn.Sigmoid()(self.conv_zt(stacked_x_hidden))
            r_t = nn.Sigmoid()(self.conv_rt(stacked_x_hidden))
            h_hat = nn.Tanh()(self.conv_h_hat(torch.cat([x_t, r_t * h_t], dim=1)))
            h_t = (1 - z_t) * h_t + z_t * h_hat

        return h_t.squeeze(0)

    def test_sizes(self, x, h_t):
        """Test for checking the input and hidden dimensions which should be the same
        """
        assert x.size(2) == h_t.size(2)
        assert x.size(3) == h_t.size(3)
