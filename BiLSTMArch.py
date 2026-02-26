import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_channels=3, hidden_size=128, num_layers=2, output_channels=24):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_channels)

    def forward(self, x):  
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0)) 
        out = self.fc(out) 

        out = out.permute(0, 2, 1).reshape(batch_size, 24, height, width)

        return out
    
model = BiLSTM()
# print(model)
input_tensor = torch.randn(8, 3, 32, 32) 
output = model(input_tensor)
print(f"Output shape: {output.shape}")
