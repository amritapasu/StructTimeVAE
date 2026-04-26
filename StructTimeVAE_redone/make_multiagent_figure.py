"""
make_multiagent_figure.py

6-panel 3x2 grid showing multiple agents across timesteps in a single scene.
Obs window = steps 0-7, pred window = steps 8-19.
"""

import sys, os
sys.path.insert(0, '/p3/home/mekeys/StructTimeVAE_redone')

import torch
import numpy as np
import cv2
import importlib.util

CKPT        = '/p3/home/mekeys/StructTimeVAE_redone/ckpt_full_v12/latest.pt'
CONFIG_PATH = '/p3/home/mekeys/StructTimeVAE_redone/config/sdd_train.py'
TEST_DIR    = '/p3/home/mekeys/SDD_embeddings/test'
CACHE_PATH  = '/p3/home/mekeys/SDD_embeddings/cache.h5'
OUTPUT_PATH = '/p3/home/mekeys/StructTimeVAE_redone/multiagent_figure.png'

TARGET_BATCH  = 12
SHOW_STEPS    = [0, 4, 7, 8, 12, 16]
OBS_LEN       = 8
HALF          = 200
GAP_H         = 10
GAP_V         = 10
LABEL_H       = 36
COLS          = 3
ANN_W, ANN_H  = 1630, 1948
VID_W, VID_H  = 904, 1080

AGENT_PALETTE = [
    (160, 80,  10),  # dark amber
    (20,  90, 200),  # dark blue
    (30, 130,  30),  # dark green
    (140, 20, 140),  # dark purple
    (20, 130, 150),  # dark teal
    (160, 60,  30),  # dark terracotta
    (70,  70, 180),  # dark periwinkle
    (170, 110, 20),  # dark gold
    (40, 150, 100),  # dark seafoam
    (160, 40,  90),  # dark rose
]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def ann2vid(x, y):
    return x * VID_W / ANN_W, y * VID_H / ANN_H


def to_displacements(x, y):
    x_pos  = x[:, :, :2]
    x_disp = torch.zeros_like(x_pos)
    x_disp[1:] = x_pos[1:] - x_pos[:-1]
    x_rel  = torch.cat([x_disp, x[:, :, 2:]], dim=-1)
    last_obs = x_pos[-1]
    anchor   = torch.cat([last_obs.unsqueeze(1), y], dim=1)
    y_rel    = anchor[:, 1:] - anchor[:, :-1]
    return x_rel, y_rel, last_obs


def to_absolute(pred_disp, last_obs):
    return torch.cumsum(pred_disp, dim=2) + last_obs.unsqueeze(0).unsqueeze(2)


def draw_pill(img, x0, y0, x1, y1, r, color):
    cv2.rectangle(img, (x0+r, y0), (x1-r, y1), color, -1)
    cv2.rectangle(img, (x0, y0+r), (x1, y1-r), color, -1)
    for cx, cy in [(x0+r,y0+r),(x1-r,y0+r),(x0+r,y1-r),(x1-r,y1-r)]:
        cv2.circle(img, (cx, cy), r, color, -1)


if __name__ == '__main__':

    spec   = importlib.util.spec_from_file_location('config', CONFIG_PATH)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from struct_time_vae import StructTimeVAE
    model = StructTimeVAE(**config.model)
    state = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    model.eval().to(device)
    print(f"Loaded checkpoint epoch {state['epoch']}")

    from data import Dataloader
    ds = Dataloader(
        [TEST_DIR], batch_first=False, device=torch.device('cpu'), seed=1,
        cache_path=CACHE_PATH, shuffle=False, **config.test_dataloader,
    )
    dl = torch.utils.data.DataLoader(
        ds, collate_fn=ds.collate_fn, batch_sampler=ds.batch_sampler
    )

    batch_data = None
    for batch_idx, item in enumerate(dl):
        if batch_idx == TARGET_BATCH:
            batch_data = item
            break
    assert batch_data is not None

    x, y, neighbor, scene_emb, seq_len, video_names, start_frames = batch_data
    scene_emb = scene_emb.to(device)
    x         = x.to(device)
    y         = y.to(device)
    if y.dim() == 4:
        y = y.squeeze(0) if y.size(0)==1 else y.squeeze(1)

    x_raw = x.clone()
    x_rel, y_rel, last_obs = to_displacements(x, y)
    x_rel[:, :, :2] = torch.clamp(x_rel[:, :, :2], -50, 50)

    with torch.no_grad():
        y_det_disp = model(x_rel, scene_emb, neighbor=neighbor, last_obs=last_obs)

    y_det_abs   = to_absolute(y_det_disp.unsqueeze(0), last_obs).squeeze(0)
    x_abs_np    = x_raw[:, :, :2].cpu().numpy()
    pred_np_all = y_det_abs.cpu().numpy()

    disps  = np.linalg.norm(x_abs_np[-1] - x_abs_np[0], axis=-1)
    moving = np.where(disps > 30)[0]
    agent_color = {int(b): AGENT_PALETTE[i % len(AGENT_PALETTE)]
                   for i, b in enumerate(moving[:10])}

    video_name  = video_names[0]
    start_frame = start_frames[0]
    mp4_path    = ds.video_map.get(video_name)
    assert mp4_path and os.path.exists(mp4_path)

    panels = []
    for step in SHOW_STEPS:
        is_pred = step >= OBS_LEN

        vid_frame = start_frame + step
        cap = cv2.VideoCapture(mp4_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_frame)
        ret, frame = cap.read()
        cap.release()
        assert ret

        # centroid of moving agents at this step
        pos_vid = []
        for b in moving[:10]:
            if step < OBS_LEN:
                ax, ay = ann2vid(*x_abs_np[step, b])
            else:
                ps = min(step - OBS_LEN, pred_np_all.shape[1]-1)
                ax, ay = ann2vid(*pred_np_all[b, ps])
            pos_vid.append((ax, ay))

        cx = int(np.mean([p[0] for p in pos_vid]))
        cy = int(np.mean([p[1] for p in pos_vid]))
        H_f, W_f = frame.shape[:2]
        x0 = max(0, min(cx - HALF, W_f - HALF*2))
        y0 = max(0, min(cy - HALF, H_f - HALF*2))
        crop = frame[y0:y0+HALF*2, x0:x0+HALF*2].copy()

        # vignette
        vign = np.ones((HALF*2, HALF*2), dtype=np.float32)
        for i in range(HALF*2):
            for j in range(HALF*2):
                dx = (i - HALF) / HALF
                dy = (j - HALF) / HALF
                vign[i, j] = max(0.72, 1.0 - 0.28*(dx**2 + dy**2))
        crop = (crop * vign[:,:,None]).astype(np.uint8)

        def to_px(ax, ay):
            vx, vy = ann2vid(ax, ay)
            return int(vx) - x0, int(vy) - y0

        def in_crop(px, py):
            return 0 <= px < HALF*2 and 0 <= py < HALF*2

        for b in moving[:10]:
            color     = agent_color[int(b)]
            obs_color = tuple(min(255, int(c * 0.55 + 110)) for c in color)

            # obs history up to this step
            obs_end = min(step, OBS_LEN - 1)
            obs_pts = [to_px(*x_abs_np[t, b]) for t in range(obs_end + 1)]
            for t in range(1, len(obs_pts)):
                p0, p1 = obs_pts[t-1], obs_pts[t]
                if in_crop(*p0) or in_crop(*p1):
                    cv2.line(crop, p0, p1, obs_color, 3, cv2.LINE_AA)

            # predicted future (draw up to current step)
            if step >= OBS_LEN - 1:
                pred_pts = [to_px(*pred_np_all[b, t])
                            for t in range(pred_np_all.shape[1])]
                pred_end = min(step - OBS_LEN + 2, len(pred_pts)) if is_pred else 1
                for t in range(1, pred_end):
                    p0, p1 = pred_pts[t-1], pred_pts[t]
                    if in_crop(*p0) or in_crop(*p1):
                        cv2.line(crop, p0, p1, color, 3, cv2.LINE_AA)

            # current position dot
            if step < OBS_LEN:
                cur_pt = to_px(*x_abs_np[step, b])
            else:
                ps = min(step - OBS_LEN, pred_np_all.shape[1]-1)
                cur_pt = to_px(*pred_np_all[b, ps])
            if in_crop(*cur_pt):
                cv2.circle(crop, cur_pt, 7, color, -1, cv2.LINE_AA)
                cv2.circle(crop, cur_pt, 7, (255,255,255), 2, cv2.LINE_AA)

        border_col = (180, 80, 20) if not is_pred else (20, 120, 220)
        cv2.rectangle(crop, (0,0), (HALF*2-1, HALF*2-1), border_col, 3)

        bar   = np.full((LABEL_H, HALF*2, 3), 255, dtype=np.uint8)
        label = f"obs  t={step}" if not is_pred else f"pred t={step}"
        draw_pill(bar, 10, 3, HALF*2-10, LABEL_H-3, 6, border_col)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.65, 2)
        cv2.putText(bar, label, ((HALF*2-tw)//2, (LABEL_H+th)//2-1),
                    FONT, 0.65, (255,255,255), 2, cv2.LINE_AA)
        panels.append(np.vstack([crop, bar]))
        print(f"  step {step} done")

    gap_h_img = np.full((panels[0].shape[0], GAP_H, 3), 255, dtype=np.uint8)
    gap_v_img = np.full((GAP_V, COLS*(HALF*2)+(COLS-1)*GAP_H, 3), 255, dtype=np.uint8)

    row1 = np.hstack([panels[0], gap_h_img, panels[1], gap_h_img, panels[2]])
    row2 = np.hstack([panels[3], gap_h_img, panels[4], gap_h_img, panels[5]])
    figure = np.vstack([row1, gap_v_img, row2])

    # Legend
    LEG_H      = 52
    SAMP       = (180, 100, 50)
    obs_samp   = tuple(min(255, int(c * 0.55 + 110)) for c in SAMP)
    leg        = np.full((LEG_H, figure.shape[1], 3), 255, dtype=np.uint8)
    cv2.line(leg, (0, LEG_H-1), (figure.shape[1], LEG_H-1), (220,220,220), 1)
    cy_leg = LEG_H // 2
    x_cur  = figure.shape[1] - 20

    for line_col, dot_col, text in reversed([
        (obs_samp, SAMP, "Observed history"),
        (SAMP,     SAMP, "Predicted trajectory"),
    ]):
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.65, 2)
        x_cur -= 30 + 8 + tw + 20
        cv2.line(leg, (x_cur, cy_leg), (x_cur+26, cy_leg), line_col, 3, cv2.LINE_AA)
        cv2.circle(leg, (x_cur+26, cy_leg), 7, dot_col, -1, cv2.LINE_AA)
        cv2.circle(leg, (x_cur+26, cy_leg), 7, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(leg, text, (x_cur+40, cy_leg+th//2),
                    FONT, 0.65, (60,60,60), 2, cv2.LINE_AA)

    PAD    = 16
    H_fig, W_fig = figure.shape[:2]
    canvas = np.full((H_fig+LEG_H+PAD*2, W_fig+PAD*2, 3), 255, dtype=np.uint8)
    canvas[PAD:PAD+LEG_H, PAD:PAD+W_fig]             = leg
    canvas[PAD+LEG_H:PAD+LEG_H+H_fig, PAD:PAD+W_fig] = figure
    cv2.imwrite(OUTPUT_PATH, canvas)
    print(f"Saved: {OUTPUT_PATH}")