import torch
import torch.nn as nn

class GRUImagePredictor(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=256, output_hours=24, feature_dim=128):
        super(GRUImagePredictor, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1),  # [B, 128, 8, 8]
            nn.ReLU()
        )

        self.spatial_size = 8 * 8
        self.gru = nn.GRU(
            input_size=feature_dim * self.spatial_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.decoder_fc = nn.Linear(hidden_dim, feature_dim * self.spatial_size)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)  # output 1 channel per hour
        )

        self.output_hours = output_hours

    def forward(self, x):
        B = x.size(0)

        feats = self.encoder(x)  
        feats_flat = feats.view(B, -1) 

        feats_seq = feats_flat.unsqueeze(1).repeat(1, self.output_hours, 1) 

        gru_out, _ = self.gru(feats_seq) 

        decoded_frames = []
        for t in range(self.output_hours):
            frame_feat = self.decoder_fc(gru_out[:, t, :]) 
            frame_feat = frame_feat.view(B, -1, 8, 8) 
            frame_img = self.decoder_conv(frame_feat) 
            decoded_frames.append(frame_img)

        output = torch.cat(decoded_frames, dim=1) 
        return output


if __name__ == "__main__":
    model = GRUImagePredictor()
    x = torch.randn(8, 3, 32, 32)  
    y = model(x)
    print(y.shape) 
