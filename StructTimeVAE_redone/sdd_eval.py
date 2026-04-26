from .sdd_train import *  # import all default SDD config

test_dataloader["traj_max_overlap"] = None          # use full trajectories for test
test_dataloader["inclusive_groups"] = ["Pedestrian"]  # only include pedestrians in evaluation