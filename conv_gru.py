import torch
import torch.nn as nn



class ConvGruCell(nn.Module):
	"""Implementation of Convolution GRU cell as described in https://arxiv.org/pdf/1511.06432.pdf
	"""

	def __init__(self, input_channels, hidden_dim, device, kernel_size=3):
		super(ConvGruCell, self).__init__()

		self.input_channels = input_channels
		self.hidden_dim = hidden_dim
		self.kernel_size = kernel_size
		self.padding = kernel_size // 3
		self.device = device

		self.conv_zt = nn.Conv2d(self.input_channels + self.hidden_dim, 
			 self.hidden_dim, padding=self.padding, kernel_size=self.kernel_size)
		self.conv_rt = nn.Conv2d(self.input_channels + self.hidden_dim, 
			 self.hidden_dim, padding=self.padding, kernel_size=self.kernel_size)
		self.conv_h_hat = nn.Conv2d(self.input_channels + self.hidden_dim, 
			 self.hidden_dim, padding=self.padding, kernel_size=self.kernel_size)
		

	def forward(self, x, h_t=None):
		"""Forward pass implementation of the ConvGruCell

		Args
			x: input tensor of shape (batch, channel, height, width)
			h_t: previous hidden vector of shape (batch_size, hidden_dim, height, width)
		"""

		if h_t is None:
			h_t = torch.zeros(x.size(0), self.hidden_dim, x.size(2) , x.size(3), device = self.device)


		self.test_sizes(x, h_t)

		stacked_x_hidden = torch.cat([x, h_t], dim=1)

		z_t = nn.Sigmoid()(self.conv_zt(stacked_x_hidden))
		r_t = nn.Sigmoid()(self.conv_rt(stacked_x_hidden))
		h_hat = nn.Tanh()(self.conv_h_hat(torch.cat([x, r_t * h_t], dim=1)))
		h_t = (1 - z_t) * h_t + z_t * h_hat

		return h_t


	def test_sizes(self, x, h_t):
		"""Test for checking the input and hidden dimensions which should be the same
		"""
		assert x.size(0) == h_t.size(0)
		assert x.size(2) == h_t.size(2)
		assert x.size(3) == h_t.size(3)
