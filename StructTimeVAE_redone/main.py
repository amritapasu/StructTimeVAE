import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from struct_time_vae import StructTimeVAE
from data import Dataloader
from utils import ADE_FDE, seed, get_rng_state, set_rng_state
from visualize_predictions import visualize_from_eval_batch


import argparse

from struct_time_vae_ablation_a import StructTimeVAE_NoScene
from struct_time_vae_ablation_b import StructTimeVAE_StaticScene

MODEL_REGISTRY = {
    "full":       StructTimeVAE,
    "ablation_a": StructTimeVAE_NoScene,
    "ablation_b": StructTimeVAE_StaticScene,
}

parser = argparse.ArgumentParser()
parser.add_argument("--train",        nargs="+", default=[])
parser.add_argument("--map_dir",      type=str,  default=None)
parser.add_argument("--test",         nargs="+", default=[])
parser.add_argument("--test_map_dir", type=str,  default=None)
parser.add_argument("--config",       type=str,  default=None)
parser.add_argument("--ckpt",         type=str,  default=None)
parser.add_argument("--seed",         type=int,  default=1)
parser.add_argument("--device",       type=str,  default=None)
parser.add_argument("--workers",      type=int,  default=1)
parser.add_argument("--rank",         type=int,  default=None)
parser.add_argument("--master_addr",  type=str,  default="localhost")
parser.add_argument("--master_port",  type=str,  default="29500")
parser.add_argument("--cache_path",   type=str,  default=None)
parser.add_argument("--model", type=str, default="full",
                    choices=["full", "ablation_a", "ablation_b"])
# NEW: visualization flags
parser.add_argument("--vis_mode",     type=str,  default="deterministic",
                    choices=["deterministic", "stochastic"],
                    help="Which prediction to visualize: deterministic or stochastic (best sample).")
parser.add_argument("--no_vis",       action="store_true",
                    help="Disable visualization even during eval.")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def expand_scene_dirs(paths):
    expanded = []
    for p in paths:
        if os.path.isdir(p):
            subdirs = sorted(
                os.path.join(p, d)
                for d in os.listdir(p)
                if os.path.isdir(os.path.join(p, d))
            )
            expanded.extend(subdirs if subdirs else [p])
        else:
            expanded.append(p)
    return expanded


def to_displacements(x, y):
    """
    x : [obs_len, B, 6]  absolute coords
    y : [B, horizon, 2]  absolute coords

    Returns:
      x_rel    : [obs_len, B, 6]
      y_rel    : [B, horizon, 2]
      last_obs : [B, 2]
    """
    x_pos  = x[:, :, :2]
    x_disp = torch.zeros_like(x_pos)
    x_disp[1:] = x_pos[1:] - x_pos[:-1]
    x_rel  = torch.cat([x_disp, x[:, :, 2:]], dim=-1)

    last_obs = x_pos[-1]
    anchor   = torch.cat([last_obs.unsqueeze(1), y], dim=1)
    y_rel    = anchor[:, 1:] - anchor[:, :-1]

    return x_rel, y_rel, last_obs


def to_absolute(pred_disp, last_obs):
    """
    pred_disp : [S, B, horizon, 2]
    last_obs  : [B, 2]
    Returns   : [S, B, horizon, 2]
    """
    abs_pos = torch.cumsum(pred_disp, dim=2)
    abs_pos = abs_pos + last_obs.unsqueeze(0).unsqueeze(2)
    return abs_pos


def squeeze_y(y):
    if y.dim() == 4:
        if y.size(0) == 1:
            y = y.squeeze(0)
        elif y.size(1) == 1:
            y = y.squeeze(1)
    return y


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    settings = parser.parse_args()
    settings.train = expand_scene_dirs(settings.train)
    settings.test  = expand_scene_dirs(settings.test)
    if not settings.test_map_dir:
        settings.test_map_dir = settings.map_dir

    import importlib
    spec   = importlib.util.spec_from_file_location(
        "config", settings.config,
        submodule_search_locations=[os.path.dirname(settings.config)],
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    device = torch.device(
        settings.device if settings.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    settings.workers = max(1, settings.workers)
    assert settings.rank is not None or settings.workers <= 1

    seed(settings.seed)
    rng_state = get_rng_state(device)

    if settings.rank is not None:
        os.environ["MASTER_ADDR"] = settings.master_addr
        os.environ["MASTER_PORT"] = settings.master_port
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        torch.set_num_threads(min(torch.get_num_threads() // device_count, 20))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    preload_kwargs = dict(
        num_workers=6, pin_memory=False, prefetch_factor=2, persistent_workers=True,
    ) if getattr(config, "preload_data", False) else {}

    loader_kwargs = dict(
        batch_first=False,
        device="cpu" if getattr(config, "preload_data", False) else device,
        seed=settings.seed,
    )

    train_data = test_data = None

    if settings.test:
        test_dataset = Dataloader(
            settings.test, **loader_kwargs,
            **config.test_dataloader,
            map_dir=settings.test_map_dir,
            cache_path=settings.cache_path,
            shuffle=False,
        )
        test_data = torch.utils.data.DataLoader(
            test_dataset,
            collate_fn=test_dataset.collate_fn,
            batch_sampler=test_dataset.batch_sampler,
            **preload_kwargs,
        )

    if settings.train:
        config.train_dataloader["batch_size"] //= settings.workers
        train_dataset = Dataloader(
            settings.train, **loader_kwargs,
            **config.train_dataloader,
            map_dir=settings.map_dir,
            cache_path=settings.cache_path,
            shuffle=True,
        )
        train_data = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=train_dataset.collate_fn,
            batch_sampler=train_dataset.batch_sampler,
            **preload_kwargs,
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    ModelClass = MODEL_REGISTRY[settings.model]
    model = ModelClass(**config.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6,
    )

    cfg_beta_agent = config.model.get("kl_beta_agent", 1.0)
    cfg_beta_scene = config.model.get("kl_beta_scene", 1.0)

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------
    if settings.ckpt:
        os.makedirs(settings.ckpt, exist_ok=True)

    start_epoch = 0
    ade_best    = float("inf")

    if settings.ckpt:
        latest_ckpt = os.path.join(settings.ckpt, "latest.pt")
        if os.path.exists(latest_ckpt):
            print(f"Loading checkpoint: {latest_ckpt}")
            state = torch.load(latest_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            rng_state   = [r.to("cpu") if torch.is_tensor(r) else r for r in state["rng_state"]]
            start_epoch = state["epoch"]
            ade_best    = state.get("ade_best", float("inf"))
            print(f"Resumed from epoch {start_epoch}, best ADE: {ade_best:.4f}")

    end_epoch = (
        start_epoch + 1
        if train_data is None or start_epoch >= config.epochs
        else config.epochs
    )

    # ------------------------------------------------------------------
    # DDP
    # ------------------------------------------------------------------
    if settings.rank is not None:
        backend    = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, rank=settings.rank, world_size=settings.workers)
        device_ids = None if settings.device is None else [device]
        model      = DDP(model, device_ids=device_ids)
        model.state_dict = model.module.state_dict
        is_chief   = settings.rank == 0
    else:
        is_chief = True

    logger = (
        SummaryWriter(log_dir=settings.ckpt)
        if is_chief and train_data is not None and settings.ckpt
        else None
    )

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch + 1, end_epoch + 1):

        anneal = min(1.0, epoch / 40.0)
        beta_a = anneal * cfg_beta_agent
        beta_s = anneal * cfg_beta_scene

        # ------------------------------------------------------------------
        # Train
        # ------------------------------------------------------------------
        trained_this_epoch = False

        if train_data is not None and epoch <= config.epochs:
            print(f"\nEpoch {epoch}/{config.epochs}  |  anneal={anneal:.3f}  beta_a={beta_a:.3f}")
            tic       = time.time()
            set_rng_state(rng_state, device)
            model.train()
            losses             = {"loss": 0.0, "rec": 0.0, "kl_agent": 0.0, "kl_scene": 0.0}
            trained_this_epoch = True

            for batch_idx, item in enumerate(train_data):

                if train_data.dataset.use_map:
                    x, y, neighbor, scene_embeddings, seq_len, _video_names, _start_frames = item
                    scene_embeddings = scene_embeddings.to(device, non_blocking=True)
                else:
                    x, y, neighbor, seq_len, _video_names, _start_frames = item
                    scene_embeddings = None

                x        = x.to(device, non_blocking=True)
                y        = y.to(device, non_blocking=True)
                neighbor = neighbor.to(device, non_blocking=True)
                seq_len  = seq_len.to(device, non_blocking=True)

                y = squeeze_y(y)

                # Debug: first batch of first epoch only
                if batch_idx == 0 and epoch == start_epoch + 1:
                    print(f"  [debug] x:{x.shape} y:{y.shape} neighbor:{neighbor.shape}")
                    if scene_embeddings is not None:
                        print(f"  [debug] scene_embeddings:{scene_embeddings.shape}")
                    print(f"  [debug] x min/max: {x.min().item():.2f}/{x.max().item():.2f}")
                    print(f"  [debug] y min/max: {y.min().item():.2f}/{y.max().item():.2f}")

                # Convert to displacements
                x, y, last_obs = to_displacements(x, y)

                # Clip extreme displacements
                y = torch.clamp(y, -50.0, 50.0)
                x[:, :, :2] = torch.clamp(x[:, :, :2], -50.0, 50.0)

                optimizer.zero_grad()
                loss_dict = model(
                    x, scene_embeddings, y,
                    neighbor=neighbor,
                    last_obs=last_obs,
                    kl_beta_agent=beta_a,
                    kl_beta_scene=beta_s,
                )

                if not torch.isfinite(loss_dict["loss"]):
                    print(f"  [warn] non-finite loss at batch {batch_idx}, skipping")
                    optimizer.zero_grad()
                    continue

                loss_dict["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                for k in losses:
                    v         = loss_dict[k].item()
                    losses[k] = (losses[k] * batch_idx + v) / (batch_idx + 1)

                if batch_idx % 20 == 0:
                    print(
                        f"  batch {batch_idx:4d}  "
                        f"loss={losses['loss']:.4f}  "
                        f"rec={losses['rec']:.4f}  "
                        f"kl_agent={losses['kl_agent']:.4f}  "
                        f"kl_scene={losses['kl_scene']:.4f}"
                    )

            rng_state = get_rng_state(device)
            print(f"  Epoch time: {time.time() - tic:.1f}s")

        # ------------------------------------------------------------------
        # Eval
        # ------------------------------------------------------------------
        ade = fde = ade_d = fde_d = ade_zero = None

        # Resolve visualization output directory: <ckpt>/vis/epoch_<N>/
        do_vis = (
            not settings.no_vis
            and settings.ckpt is not None
            and test_data is not None
            and test_data.dataset.use_map
            and is_chief
        )
        vis_dir = (
            os.path.join(settings.ckpt, "vis", f"epoch_{epoch:04d}")
            if do_vis else None
        )
        if do_vis:
            os.makedirs(vis_dir, exist_ok=True)

        if test_data is not None:
            model.eval()
            ADE, FDE, ADE_d, FDE_d, ADE_zero, ADE_moving = [], [], [], [], [], []

            n_samples = getattr(config, "pred_samples", 20)

            with torch.no_grad():
                for batch_idx, item in enumerate(test_data):

                    if test_data.dataset.use_map:
                        x, y, neighbor, scene_embeddings, seq_len, video_names, start_frames = item
                        scene_embeddings = scene_embeddings.to(device, non_blocking=True)
                    else:
                        x, y, neighbor, seq_len, video_names, start_frames = item
                        scene_embeddings = None

                    x        = x.to(device, non_blocking=True)
                    y        = y.to(device, non_blocking=True)
                    neighbor = neighbor.to(device, non_blocking=True)

                    y = squeeze_y(y)

                    # Save raw absolute positions before displacement conversion
                    x_raw = x.clone()  # [obs_len, B, 6] absolute pixel coords

                    x, y_rel, last_obs = to_displacements(x, y)
                    y_abs = y
                    x[:, :, :2] = torch.clamp(x[:, :, :2], -50.0, 50.0)

                    # Zero-velocity baseline
                    y_zero_abs = last_obs.unsqueeze(1).expand_as(y_abs)
                    ade_zero_b, _ = ADE_FDE(y_zero_abs.unsqueeze(0), y_abs)
                    ADE_zero.append(ade_zero_b.squeeze(0))

                    # Stochastic
                    y_samp_disp = model.sample(
                        x, scene_embeddings,
                        n_samples=n_samples,
                        neighbor=neighbor,
                        last_obs=last_obs,
                    )
                    y_samp_abs  = to_absolute(y_samp_disp, last_obs)
                    ade_b, fde_b = ADE_FDE(y_samp_abs, y_abs)
                    ADE.append(ade_b.min(dim=0)[0])
                    FDE.append(fde_b.min(dim=0)[0])

                    # Moving-agent ADE
                    total_disp = (y_abs[:, -1, :] - y_abs[:, 0, :]).norm(dim=-1)
                    moving = total_disp > 20.0
                    if moving.any():
                        ade_moving_b, _ = ADE_FDE(y_samp_abs[:, moving, :, :], y_abs[moving])
                        ADE_moving.append(ade_moving_b.min(dim=0)[0])

                    # Deterministic
                    y_det_disp = model(
                        x, scene_embeddings,
                        neighbor=neighbor,
                        last_obs=last_obs,
                    )
                    y_det_abs = to_absolute(y_det_disp.unsqueeze(0), last_obs).squeeze(0)
                    ade_d_b, fde_d_b = ADE_FDE(y_det_abs.unsqueeze(0), y_abs)
                    ADE_d.append(ade_d_b.squeeze(0))
                    FDE_d.append(fde_d_b.squeeze(0))

                    # ----------------------------------------------------------
                    # Visualization: one MP4 per unique video in this batch.
                    # Multiple batch items from the same video are written to
                    # the same file (the first item encountered wins — subsequent
                    # items with the same video name are skipped to avoid
                    # overwriting with a different temporal window).
                    # ----------------------------------------------------------
                    if do_vis:
                        frameskip   = getattr(test_data.dataset, "frameskip", 1)
                        written_vis = set()  # track videos already written this batch

                        for b_idx, (video_name, start_frame) in enumerate(
                            zip(video_names, start_frames)
                        ):
                            if video_name is None:
                                continue
                            if video_name in written_vis:
                                continue  # already wrote a clip for this video this batch

                            mp4_path = test_data.dataset.video_map.get(video_name)
                            if mp4_path is None or not os.path.exists(mp4_path):
                                continue

                            # DEBUG: skip stationary agents
                            # DEBUG: target item 29 agent
                            _pos0 = x_raw[0, b_idx, :2].cpu().numpy()
                            if not (430 < float(_pos0[0]) < 470 and 1080 < float(_pos0[1]) < 1110):
                                continue


                            # Output path: <vis_dir>/<video_name>_batch<N>.mp4
                            # Include batch index so repeated windows from the
                            # same video across different batches don't collide.
                            out_name = f"{video_name}_batch{batch_idx:04d}.mp4"
                            out_path = os.path.join(vis_dir, out_name)

                            # Slice tensors to this single agent (b_idx)
                            x_raw_b    = x_raw[:, b_idx:b_idx+1, :]       # [obs_len, 1, 6]
                            last_obs_b = last_obs[b_idx:b_idx+1]           # [1, 2]
                            y_det_b    = y_det_disp[b_idx:b_idx+1]         # [1, H, 2]
                            y_samp_b   = y_samp_disp[:, b_idx:b_idx+1, :, :]  # [K, 1, H, 2]

                            _annot_path = os.path.join(os.path.dirname(mp4_path), "annotations.txt")
                            try:
                                visualize_from_eval_batch(
                                    mp4_path    = mp4_path,
                                    x           = x_raw_b,
                                    y_det_disp  = y_det_b,
                                    last_obs    = last_obs_b,
                                    start_frame = start_frame,
                                    output_path = out_path,
                                    mode        = settings.vis_mode,
                                    y_samp_disp = y_samp_b if settings.vis_mode == "stochastic" else None,
                                    frameskip   = frameskip,
                                    y_abs       = y_abs[b_idx:b_idx+1],
                                    annot_path  = _annot_path,
                                )
                                written_vis.add(video_name)

                                # DEBUG: print coords then exit after first clip
                                obs_xy  = x_raw_b[:, 0, :2].cpu().numpy()
                                pred_xy = to_absolute(
                                    y_det_b.unsqueeze(0), last_obs_b
                                ).squeeze(0).squeeze(0).cpu().numpy()
                                print(f"\n[DEBUG] video={video_name}  start_frame={start_frame}")
                                print(f"  observed ({len(obs_xy)} steps, raw annotation coords):")
                                print(f"  seq_len for this agent: {seq_len[b_idx].item()}")
                                print(f"  x_raw_b full (all 6 features):")
                                for _t, _row in enumerate(x_raw_b[:, 0, :].cpu().numpy()):
                                    print(f"    t={_t:3d}  {_row}")
                                for t, (px, py) in enumerate(obs_xy):
                                    print(f"    t={t:3d}  x={px:.1f}  y={py:.1f}")
                                print(f"  predicted ({len(pred_xy)} steps):")
                                for t, (px, py) in enumerate(pred_xy):
                                    print(f"    t={t:3d}  x={px:.1f}  y={py:.1f}")
                                print(f"[DEBUG] output -> {out_path}\n")
                                import sys; sys.exit(0)
                            except Exception as e:
                                print(f"  [vis] WARNING: failed for {video_name}: {e}")

            ade      = torch.cat(ADE).mean().item()
            fde      = torch.cat(FDE).mean().item()
            ade_d    = torch.cat(ADE_d).mean().item()
            fde_d    = torch.cat(FDE_d).mean().item()
            ade_zero = torch.cat(ADE_zero).mean().item()

            print(
                f"  Eval  ADE(stoch/det)={ade:.4f}/{ade_d:.4f}  "
                f"FDE(stoch/det)={fde:.4f}/{fde_d:.4f}"
            )
            print(
                f"  Zero-vel baseline ADE={ade_zero:.4f}  "
                f"model vs baseline: {'BETTER' if ade < ade_zero else 'WORSE'} "
                f"({ade_zero - ade:+.4f})"
            )

            total_disp = (y_abs[:, -1, :] - y_abs[:, 0, :]).norm(dim=-1)
            n_stationary = (total_disp < 20.0).sum().item()
            n_total = total_disp.shape[0]
            print(f"stationary (<20px): {n_stationary}/{n_total} ({100*n_stationary/n_total:.1f}%)")

            if ADE_moving:
                ade_moving = torch.cat(ADE_moving).mean().item()
                print(f"  Moving-agent ADE (stoch, >20px): {ade_moving:.4f}")
            else:
                print("  No moving agents found in test set")

            if do_vis:
                print(f"  Visualizations written to {vis_dir}")

            scheduler.step(ade)

        # ------------------------------------------------------------------
        # Logging + checkpointing
        # ------------------------------------------------------------------
        if is_chief and settings.ckpt and trained_this_epoch:

            if logger is not None:
                for k, v in losses.items():
                    logger.add_scalar(f"train/{k}", v, epoch)
                logger.add_scalar("train/anneal",  anneal, epoch)
                logger.add_scalar("train/beta_a",  beta_a, epoch)
                logger.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
                if ade is not None:
                    logger.add_scalars("eval", {
                        "ADE_stoch":   ade,
                        "FDE_stoch":   fde,
                        "ADE_det":     ade_d,
                        "FDE_det":     fde_d,
                        "ADE_zeroval": ade_zero,
                    }, epoch)

            latest_ckpt = os.path.join(settings.ckpt, "latest.pt")
            torch.save({
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": rng_state,
                "epoch":     epoch,
                "ade_best":  ade_best,
            }, latest_ckpt)
            print(f"  Checkpoint saved: {latest_ckpt}")

            if ade is not None and ade < ade_best:
                ade_best  = ade
                best_ckpt = os.path.join(settings.ckpt, "best_ADE.pt")
                torch.save({
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng_state": rng_state,
                    "epoch":     epoch,
                    "ADE":       ade,
                    "ade_best":  ade_best,
                }, best_ckpt)
                print(f"  New best ADE: {ade:.4f} -> {best_ckpt}")

    if logger is not None:
        logger.close()