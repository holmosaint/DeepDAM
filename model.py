import torch
import torch.nn as nn
import sys

from arch.resnetgn import resnet18gn, resnet34gn, resnet50gn, resnet101gn, resnet152gn

class FeatureExtractor(nn.Module):

    def __init__(
        self,
        time_len,
        sequence_num,
        output_dim,
        cnn_type,
        time_type,
        fc_layers,
        cnn_config,
        cuda=True,
        dropout=False,
    ):
        """
        Initializes the FeatureExtractor class.

        Args:
            time_len (int): The length of the time series data.
            sequence_num (int): The number of sequences.
            output_dim (int): The dimension of the output.
            cnn_type (str): The type of CNN to use. Currently supports 'resnet'.
            time_type (str): The type of time series model to use. Currently supports 'lstm'.
            fc_layers (int): The number of fully connected layers.
            cnn_config (int): The configuration of the CNN. Supports [18, 34, 50, 101, 152] for ResNet.
            cuda (bool, optional): Whether to use CUDA. Defaults to True.
            dropout (bool, optional): Whether to use dropout. Defaults to False.

        Raises:
            NotImplementedError: If the cnn_type or time_type is not supported.
            SystemExit: If the cnn_config is not supported for ResNet.
        """
        super(FeatureExtractor, self).__init__()
        self.time_len = time_len
        self.sequence_num = sequence_num
        self.output_dim = output_dim
        self.cnn_type = cnn_type
        self.time_type = time_type
        self.fc_layers = fc_layers
        self._cuda = cuda
        self.dropout = dropout
        self.cnn_config = cnn_config

        if self.cnn_type == 'resnet':
            if self.cnn_config == 18:
                self.conv = resnet18gn(self.sequence_num)
            elif self.cnn_config == 34:
                self.conv = resnet34gn(self.sequence_num)
            elif self.cnn_config == 50:
                self.conv = resnet50gn(self.sequence_num)
            elif self.cnn_config == 101:
                self.conv = resnet101gn(self.sequence_num)
            elif self.cnn_config == 152:
                self.conv = resnet152gn(self.sequence_num)
            else:
                print(
                    "Current ResNet arch also supports layer: [18, 34, 50, 101, 152], but got {}"
                    .format(self.cnn_config))
                sys.exit(-1)

            resdim = {18: 512, 34: 2048, 50: 2048, 101: 2048, 152: 2048}
            conv_last_dim = resdim[cnn_config]
        else:
            raise NotImplementedError()

        self.time_hid_dim = conv_last_dim
        if self.time_type == 'lstm':
            self.time_net = nn.LSTM(self.time_hid_dim,
                                    self.time_hid_dim,
                                    bias=True)
        else:
            raise NotImplementedError()

        self.conv1x1 = nn.Sequential(*[
            nn.Conv1d(self.time_hid_dim, 512, kernel_size=1, bias=True),
            nn.LeakyReLU(inplace=False),
            nn.GroupNorm(32, 512),
            nn.AvgPool1d(4, stride=1)   
        ])

        with torch.no_grad():
            x = torch.zeros(1, self.sequence_num, self.time_len)
            x = self.get_feature(x)
            self.fc_dim = x.view(1, -1).shape[1]

    def init_hidden(self, hidden_dim, batch, layer, device):
        """
        Initializes the hidden state for an RNN.

        Parameters:
        hidden_dim (int): The number of features in the hidden state.
        batch (int): The batch size.
        layer (int): The number of recurrent layers.

        Returns:
        torch.nn.Parameter or tuple: Initialized hidden state. If the RNN type is LSTM, 
                         returns a tuple of (hidden state, cell state).
        """
        document_rnn_init_h = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(layer, batch, hidden_dim).type(torch.FloatTensor)),
                                           requires_grad=True)

        if self.time_type == 'lstm':
            document_rnn_init_c = nn.Parameter(nn.init.xavier_uniform_(
                torch.Tensor(layer, batch, hidden_dim).type(torch.FloatTensor)),
                                               requires_grad=True)
            return (document_rnn_init_h.to(device), document_rnn_init_c.to(device))

    def forward(self, dynamic):
        """
        Forward pass for the model.

        Parameters:
        dynamic (torch.Tensor): The dynamic input tensor with the last dimension equal to `self.time_len`.
        feature (torch.Tensor, optional): Precomputed feature tensor. If None, it will be computed using `get_feature`.

        Returns:
        torch.Tensor: Flattened feature tensor starting from dimension 1 to the last dimension.
        """
        assert dynamic.shape[-1] == self.time_len
        feature = self.get_feature(dynamic)

        return torch.flatten(feature, start_dim=1, end_dim=-1)

    def get_feature(self, dynamic):
        batch = dynamic.shape[0]
        if len(dynamic.shape) == 2:
            dynamic = dynamic.unsqueeze(1)
        assert dynamic.shape[
            1] == self.sequence_num, "Dynamic shape: {} seq num: {}".format(
                dynamic.shape, self.sequence_num)
        x = self.conv(dynamic)
        x = x.permute((2, 0, 1))
        x = self.time_net(x, self.init_hidden(self.time_hid_dim, batch, 1, x.device))
        if type(x) is tuple:
            x = x[0]
        x = x.permute((1, 2, 0))
        x = self.conv1x1(x)

        return x


class Classifier(nn.Module):

    def __init__(self, in_dim, output_dim, fc_layers, cuda=True, dropout=False):
        """
        Initializes the Classifier model.

        Args:
            in_dim (int): The dimension of the input features.
            output_dim (int): The dimension of the output features.
            fc_layers (int): The number of fully connected layers. Must be between 0 and 5.
            cuda (bool, optional): If True, use CUDA for computation. Default is True.
            dropout (bool, optional): If True, apply dropout with a probability of 0.1 after each fully connected layer. Default is False.

        Raises:
            AssertionError: If fc_layers is not between 0 and 5.
        """
        super(Classifier, self).__init__()

        self.in_dim = in_dim
        self.output_dim = output_dim
        self.fc_layers = fc_layers
        self._cuda = cuda
        self.dropout = dropout

        fc_list = list()
        fc_dim_list = [in_dim]

        assert self.fc_layers <= 5 and self.fc_layers >= 0

        for i in range(self.fc_layers):
            fc = nn.Sequential(*[
                nn.Linear(fc_dim_list[i], fc_dim_list[i + 1]),
                nn.Dropout(0.1) if self.dropout else nn.Identity(),
                nn.LeakyReLU(inplace=False)
            ])
            fc_list.append(fc)
        fc_list.append(nn.Linear(fc_dim_list[self.fc_layers], self.output_dim))

        self.fc = nn.ModuleList(fc_list)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor after passing through all fully connected layers.
                - list: A list of intermediate tensors after each fully connected layer.
        """
        fc_feature = list()
        for i in range(len(self.fc)):
            x = self.fc[i](x)
            fc_feature.append(x)
        return x, fc_feature
