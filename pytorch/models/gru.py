import torch.nn as nn
import torch

class TorchBiGRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, h2, layer_dim, output_dim, dropout_prob):
        super(TorchBiGRU, self).__init__()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob, bidirectional=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

class TorchGRU(nn.Module):

        def __init__(self, input_dim, hidden_dim, h2, layer_dim, output_dim, dropout_prob):
            super(TorchGRU, self).__init__()

            self.layer_dim = layer_dim
            self.hidden_dim = hidden_dim

            # GRU layers
            self.gru = nn.GRU(
                input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob, bidirectional=True
            )
            # Fully connected layer
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

        def forward(self, x):
            # Initializing hidden state for first input with zeros
            h0 = torch.zeros(self.layer_dim , x.size(0), self.hidden_dim, device=x.device).requires_grad_()

            # Forward propagation by passing in the input and hidden state into the model
            out, _ = self.gru(x, h0.detach())

            # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
            # so that it can fit into the fully connected layer
            out = out[:, -1, :]

            # Convert the final state to our desired output shape (batch_size, output_dim)
            out = self.fc(out)

            return out