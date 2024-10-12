from modules.view_finder.abstract_view_finder import ViewFinder
from modules.view_finder.gf_view_finder import GradientFieldViewFinder


class ViewFinderFactory:
    @staticmethod
    def create(name, config) -> ViewFinder:
        general_config = config["general"]
        view_finder_config = config["view_finder"][name]
        if name == "gradient_field":
            return GradientFieldViewFinder(
                pose_mode=view_finder_config["pose_mode"],
                regression_head=view_finder_config["regression_head"],
                per_point_feature=general_config["per_point_feature"],
                sample_mode=view_finder_config["sample_mode"],
                sampling_steps=view_finder_config.get("sampling_steps", None),
                sde_mode=view_finder_config["sde_mode"]
            )
        else:
            raise ValueError(f"Unknown next-best-view finder name: {name}")


''' ------------ Debug ------------ '''
if __name__ == "__main__":
    from configs.config import ConfigManager
    import torch

    ConfigManager.load_config_with('../../configs/local_train_config.yaml')
    ConfigManager.print_config()
    view_finder = ViewFinderFactory.create(name="gradient_field", config=ConfigManager.get("modules"))
    test_scene_feat = torch.rand(32, 1024).to("cuda:0")
    test_target_feat = torch.rand(32, 1024).to("cuda:0")
    test_pose = torch.rand(32, 6).to("cuda:0")
    test_t = torch.rand(32, 1).to("cuda:0")
    view_finder = view_finder.to("cuda:0")
    test_data = {
        'target_feat': test_target_feat,
        'scene_feat': test_scene_feat,
        'sampled_pose': test_pose,
        't': test_t
    }
    score = view_finder(test_data)
    print(score.shape)
    pose_6d = view_finder.next_best_view(scene_pts_feat=test_data["scene_feat"], target_pts_feat=test_data["target_feat"])
    print(pose_6d.shape)