import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print "Loading ResNet ArcFace"
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
        )
        self.facenet.load_state_dict(torch.load(model_paths["ir_se50"]))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, vector_one):
        vector_one = vector_one[:, :, 35:223, 32:220]  # Crop interesting region
        vector_one = self.face_pool(vector_one)
        x_feats = self.facenet(vector_one)
        return x_feats

    def forward(self, y_hat, y_one, x_one):
        n_samples = x_one.shape[0]
        x_feats = self.extract_feats(x_one)
        y_feats = self.extract_feats(y_one)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            id_logs.append(
                {
                    "diff_target": float(y_hat_feats[i].dot(y_feats[i])),
                    "diff_input": float(y_hat_feats[i].dot(x_feats[i])),
                    "diff_views": float(y_feats[i].dot(x_feats[i])),
                }
            )
            loss += 1 - y_hat_feats[i].dot(y_feats[i])
            id_diff = float(y_hat_feats[i].dot(y_feats[i])) - float(y_feats[i].dot(x_feats[i]))
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs
