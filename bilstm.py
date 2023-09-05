import torch
import torch.nn as nn

class BiLstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        output, (h, c) = self.lstm(x, (h_0, c_0))
        out = output[:, -1, :]

        out = self.fc(out)
        return out