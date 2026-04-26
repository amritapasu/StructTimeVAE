OB_HORIZON   = 8
PRED_HORIZON = 12
OB_RADIUS    = 20
MAP_SIZE     = 128

lr           = 3e-4
epochs       = 100
test_since   = 20
preload_data = False
pred_samples = 50
clustering   = 0

train_dataloader = dict(
    ob_horizon        = OB_HORIZON,
    map_size          = MAP_SIZE,
    pred_horizon      = PRED_HORIZON,
    ob_radius         = OB_RADIUS,
    inclusive_groups  = ["pedestrian"],
    batch_size        = 128,
    batches_per_epoch = 200,
    frameskip         = 12,
    traj_max_overlap  = 1,
)

test_dataloader = dict(
    ob_horizon       = OB_HORIZON,
    map_size         = MAP_SIZE,
    pred_horizon     = PRED_HORIZON,
    ob_radius        = OB_RADIUS,
    inclusive_groups = ["pedestrian"],
    traj_max_overlap = 1,
    batch_size       = 128,
    frameskip        = 12,
)

model = dict(
    horizon        = PRED_HORIZON,
    hidden_dim     = 256,
    z_agent_dim    = 32,
    scene_dim      = 32,
    dino_embed_dim = 768,
    kl_beta_agent  = 0.5,
    free_bits      = 0.01,
)
