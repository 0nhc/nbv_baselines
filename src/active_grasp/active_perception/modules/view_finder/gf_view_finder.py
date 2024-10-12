import torch
import torch.nn as nn
from utils.pose_util import PoseUtil
from modules.view_finder.abstract_view_finder import ViewFinder
import modules.module_lib as mlib
import modules.func_lib as flib


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GradientFieldViewFinder(ViewFinder):
    def __init__(self, pose_mode='rot_matrix', regression_head='Rx_Ry', per_point_feature=False,
                 sample_mode="ode", sampling_steps=None, sde_mode="ve"):

        super(GradientFieldViewFinder, self).__init__()
        self.regression_head = regression_head
        self.per_point_feature = per_point_feature
        self.act = nn.ReLU(True)
        self.sample_mode = sample_mode
        self.pose_mode = pose_mode
        pose_dim = PoseUtil.get_pose_dim(pose_mode)
        self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps, self.T = flib.init_sde(sde_mode)
        self.sampling_steps = sampling_steps

        ''' encode pose '''
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )

        ''' encode t '''
        self.t_encoder = nn.Sequential(
            mlib.GaussianFourierProjection(embed_dim=128),
            nn.Linear(128, 128),
            self.act,
        )

        ''' fusion tail '''
        if self.regression_head == 'Rx_Ry':
            if pose_mode != 'rot_matrix':
                raise NotImplementedError
            if not per_point_feature:
                ''' rotation_x_axis regress head '''
                self.fusion_tail_rot_x = nn.Sequential(
                    nn.Linear(128 + 256 + 1024 + 1024, 256),
                    self.act,
                    zero_module(nn.Linear(256, 3)),
                )
                self.fusion_tail_rot_y = nn.Sequential(
                    nn.Linear(128 + 256 + 1024 + 1024, 256),
                    self.act,
                    zero_module(nn.Linear(256, 3)),
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, data):
        """
        Args:
            data, dict {
                'target_pts_feat': [bs, c]
                'scene_pts_feat': [bs, c]
                'pose_sample': [bs, pose_dim]
                't': [bs, 1]
            }
        """

        scene_pts_feat = data['scene_feat']
        target_pts_feat = data['target_feat']
        sampled_pose = data['sampled_pose']
        t = data['t']
        t_feat = self.t_encoder(t.squeeze(1))
        pose_feat = self.pose_encoder(sampled_pose)

        if self.per_point_feature:
            raise NotImplementedError
        else:
            total_feat = torch.cat([scene_pts_feat, target_pts_feat, t_feat, pose_feat], dim=-1)
        _, std = self.marginal_prob_fn(total_feat, t)

        if self.regression_head == 'Rx_Ry':
            rot_x = self.fusion_tail_rot_x(total_feat)
            rot_y = self.fusion_tail_rot_y(total_feat)
            out_score = torch.cat([rot_x, rot_y], dim=-1) / (std + 1e-7)  # normalisation
        else:
            raise NotImplementedError

        return out_score

    def marginal_prob(self, x, t):
        return self.marginal_prob_fn(x,t)

    def sample(self, data, atol=1e-5, rtol=1e-5, snr=0.16, denoise=True, init_x=None, T0=None):

        if self.sample_mode == 'pc':
            in_process_sample, res = flib.cond_pc_sampler(
                score_model=self,
                data=data,
                prior=self.prior_fn,
                sde_coeff=self.sde_fn,
                num_steps=self.sampling_steps,
                snr=snr,
                eps=self.sampling_eps,
                pose_mode=self.pose_mode,
                init_x=init_x
            )

        elif self.sample_mode == 'ode':
            T0 = self.T if T0 is None else T0
            in_process_sample, res = flib.cond_ode_sampler(
                score_model=self,
                data=data,
                prior=self.prior_fn,
                sde_coeff=self.sde_fn,
                atol=atol,
                rtol=rtol,
                eps=self.sampling_eps,
                T=T0,
                num_steps=self.sampling_steps,
                pose_mode=self.pose_mode,
                denoise=denoise,
                init_x=init_x
            )
        else:
            raise NotImplementedError

        return in_process_sample, res

    def next_best_view(self, scene_pts_feat, target_pts_feat):
        data = {
            'scene_feat': scene_pts_feat,
            'target_feat': target_pts_feat,
        }
        in_process_sample, res = self.sample(data)
        return res.to(dtype=torch.float32), in_process_sample


''' ----------- DEBUG -----------'''
if __name__ == "__main__":
    test_scene_feat = torch.rand(32, 1024).to("cuda:0")
    test_target_feat = torch.rand(32, 1024).to("cuda:0")
    test_pose = torch.rand(32, 6).to("cuda:0")
    test_t = torch.rand(32, 1).to("cuda:0")
    view_finder = GradientFieldViewFinder().to("cuda:0")
    test_data = {
        'target_feat': test_target_feat,
        'scene_feat': test_scene_feat,
        'sampled_pose': test_pose,
        't': test_t
    }
    score = view_finder(test_data)

    result = view_finder.next_best_view(test_scene_feat, test_target_feat)
    print(result)
