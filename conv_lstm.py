import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
	def __init__(self, input_channels, hidden_dim, seq_len,device, kernel_size=3):
		super(ConvLSTMCell, self).__init__()

		self.input_channels = input_channels
		self.hidden_dim = hidden_dim
		self.seq_len = seq_len
		self.kernel_size = kernel_size
		self.padding = kernel_size // 3
		self.device = device

		self.conv = nn.Conv2d(self.input_channels + self.hidden_dim, 
			4 * self.hidden_dim, padding=self.padding, kernel_size=self.kernel_size)

	def forward(self, x, h_t=None, c_t=None):
		"""Forward step of the convLSTMCell

		Args:
			x: input tensor of shape (batch_size, channels, height, width)
			h_t: previous hidden vector of shape (batch_size, hidden_dim, height, width)
			c_t: previous cell state of shape (batch_size, hidden_dim, height, width)
		"""
		#print("Size: ", x.size())
		if(h_t is None and c_t is None):
			h_t, c_t = self.get_hidden_state(x.size(0), x.size(2), x.size(3))
		elif( (h_t is not None and c_t is None) or (h_t is None and c_t is not None) ):
			raise RuntimeError()


		input_hidden = torch.cat([x, h_t], dim=1)
		#print("Hidden ", input_hidden.size())
		conv_output = self.conv(input_hidden)  

		#print("conv_output " , conv_output.size())

		i_cell, f_cell, o_cell, g_cell = torch.split(conv_output, self.hidden_dim, dim=1)

		i_t = nn.Sigmoid()(i_cell)
		f_t = nn.Sigmoid()(f_cell)
		o_t = nn.Sigmoid()(o_cell)
		g_t = nn.Tanh()(g_cell) 

		# hidden and Cell state for the next iteration
		c_t = f_t * c_t + i_t * g_t
		h_t = o_t * nn.Tanh()(c_t)

		#print("h_t - > " , h_t.size())
			
		return h_t, c_t


	def get_hidden_state(self, batch_size, height, width):
		h = torch.zeros(batch_size, self.hidden_dim, height, width, device = self.device)
		c = torch.zeros(batch_size, self.hidden_dim, height, width, device = self.device)

		return h, c

