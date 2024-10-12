import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    def __init__(self, rgb_dim, pts_dim, output_dim):
        super(FeatureFusion, self).__init__()
        self.pts_embedding = nn.Linear(pts_dim, output_dim)
        

        # B * patch_size * patch_size * C => B * 1 * 1 * C => B * C
        self.rgb_embedding = nn.Sequential(
            nn.Conv2d(rgb_dim, 512, kernel_size=3, stride=2, padding=1), # Bx17x17x512
            nn.ReLU(),
            nn.Conv2d(512, output_dim, kernel_size=3, stride=2, padding=1), # # Bx9x9xoutput_dim
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1), # Bx5x5xoutput_dim
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=5, stride=1, padding=0), # Bx1x1xoutput_dim
            nn.ReLU()   
        )
        self.fc_fusion = nn.Linear(output_dim * 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, img_feat, pts_feat):
        # img_feat = torch.mean(img_feat, dim=1)
        patch_length = img_feat.size(1)
        patch_size = int(patch_length ** 0.5) 
        # B * patch_size * patch_size * C = > B * C * patch_size * patch_size
        img_feat = img_feat.view(-1, patch_size, patch_size, img_feat.size(2))
        img_feat = img_feat.permute(0, 3, 2, 1)
        rgb_embedding = self.rgb_embedding(img_feat)
        rgb_embedding = rgb_embedding.view(rgb_embedding.size(0), -1)
        pts_embedding = self.relu(self.pts_embedding(pts_feat))
        fusion_feat = torch.cat((rgb_embedding, pts_embedding), dim=1)
        output = self.fc_fusion(fusion_feat)
        return output

if __name__ == "__main__":
    B = 64
    C = 1024
    img_feat_dim = 384
    pts_feat_dim = 1024
    img_feat = torch.randn(B, 1156, 384).cuda()
    pts_feat = torch.randn(B, 1024).cuda()
    fusion_model = FeatureFusion(img_feat_dim,pts_feat_dim,output_dim=C).cuda()
    output = fusion_model(img_feat, pts_feat)
    print(output.shape)