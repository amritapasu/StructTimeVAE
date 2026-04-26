"""
visualize_predictions.py

Renders predicted agent trajectories over the source video for the full
observation + prediction timeframe and writes an MP4.

Usage (deterministic):
    visualize_predictions(
        mp4_path       = "path/to/video.mp4",
        pred_abs       = y_det_abs,          # [B, horizon, 2]  absolute pixel coords
        last_obs_pos   = last_obs,           # [B, 2]           last observed pixel position
        x_abs          = x_abs,              # [obs_len, B, 2]  absolute pixel positions (obs window)
        start_frame    = start_frame,        # int              first annotation frame index
        frameskip      = 1,
        output_path    = "out_det.mp4",
        mode           = "deterministic",
    )

Usage (stochastic — picks sample with lowest ADE to deterministic pred):
    visualize_predictions(
        mp4_path       = "path/to/video.mp4",
        pred_abs       = y_samp_abs,         # [K, B, horizon, 2] absolute pixel coords
        last_obs_pos   = last_obs,           # [B, 2]
        x_abs          = x_abs,              # [obs_len, B, 2]
        start_frame    = start_frame,
        frameskip      = 1,
        output_path    = "out_stoch.mp4",
        mode           = "stochastic",
        det_pred_abs   = y_det_abs,          # [B, horizon, 2]  required for sample selection
    )

Notes on coordinate convention:
    - All positions must be in absolute pixel coordinates on the raw video frame.
    - x_abs  : the absolute positions extracted from the raw `x` tensor BEFORE
                to_displacements() is called, i.e. x[:, :, :2] in pixel space.
    - pred_abs: output of to_absolute(pred_disp, last_obs) from main.py.
    - The visualizer does NOT perform any coordinate transforms.
"""

from typing import Optional
import os

import cv2
import numpy as np
import torch

ANN_W, ANN_H = 1630, 1948  # SDD annotation coordinate space


# ---------------------------------------------------------------------------
# Colour palette — one distinct colour per agent, BGR for OpenCV
# ---------------------------------------------------------------------------
_PALETTE_BGR = [
    (  0, 200, 255),  # amber-yellow
    ( 50, 205,  50),  # lime green
    (255,  50,  50),  # blue
    (  0,  50, 255),  # red
    (255,   0, 200),  # magenta
    (  0, 165, 255),  # orange
    (255, 255,   0),  # cyan
    (147,  20, 255),  # deep pink
    ( 60, 180,  75),  # medium green
    (128,   0, 128),  # purple
]


def _agent_color(agent_idx: int):
    return _PALETTE_BGR[agent_idx % len(_PALETTE_BGR)]


# ---------------------------------------------------------------------------
# Core drawing helpers
# ---------------------------------------------------------------------------

def _draw_trajectory(
    frame:      np.ndarray,
    positions:  np.ndarray,   # [T, 2]  float pixel coords
    color_bgr:  tuple,
    thickness:  int   = 2,
    alpha:      float = 1.0,
) -> np.ndarray:
    """Draw a polyline on `frame`. Returns frame (modified in place if alpha==1)."""
    valid = np.isfinite(positions).all(axis=1) & (positions < 1e8).all(axis=1)
    pts = positions[valid].astype(np.int32)
    if len(pts) < 2:
        return frame

    if alpha < 1.0:
        overlay = frame.copy()
        for i in range(len(pts) - 1):
            cv2.line(overlay, tuple(pts[i]), tuple(pts[i + 1]), color_bgr, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    else:
        for i in range(len(pts) - 1):
            cv2.line(frame, tuple(pts[i]), tuple(pts[i + 1]), color_bgr, thickness, cv2.LINE_AA)
    return frame


def _draw_dot(
    frame:     np.ndarray,
    pos:       np.ndarray,   # [2,] float pixel coords
    color_bgr: tuple,
    radius:    int = 5,
) -> None:
    if not (np.isfinite(pos).all() and (pos < 1e8).all()):
        return
    cv2.circle(frame, tuple(pos.astype(np.int32)), radius, color_bgr, -1, cv2.LINE_AA)
    cv2.circle(frame, tuple(pos.astype(np.int32)), radius, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Select most-likely stochastic sample per agent
# (lowest ADE to deterministic prediction)
# ---------------------------------------------------------------------------

def _select_best_samples(
    y_samp_abs: np.ndarray,   # [K, B, horizon, 2]
    y_det_abs:  np.ndarray,   # [B, horizon, 2]
) -> np.ndarray:              # [B, horizon, 2]
    K, B, H, _ = y_samp_abs.shape
    # ADE: mean L2 distance over horizon for each sample/agent
    diff = y_samp_abs - y_det_abs[None]           # [K, B, H, 2]
    ade  = np.linalg.norm(diff, axis=-1).mean(-1)  # [K, B]
    best = ade.argmin(axis=0)                      # [B]
    return y_samp_abs[best, np.arange(B)]          # [B, H, 2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_predictions(
    mp4_path:       str,
    pred_abs:       "torch.Tensor | np.ndarray",  # [B, H, 2] or [K, B, H, 2]
    last_obs_pos:   "torch.Tensor | np.ndarray",  # [B, 2]
    x_abs:          "torch.Tensor | np.ndarray",  # [obs_len, B, 2]
    start_frame:    int,
    output_path:    str,
    mode:           str                         = "deterministic",  # "deterministic" | "stochastic"
    det_pred_abs:   "Optional[torch.Tensor | np.ndarray]" = None,  # [B, H, 2], required for stochastic
    frameskip:      int                         = 1,
    fps:            Optional[float]             = None,
    line_thickness: int                         = 2,
    dot_radius:     int                         = 5,
    gt_abs:         "Optional[np.ndarray]"      = None,  # [obs+horizon, B, 2] ground truth all steps
    annot_path:     "Optional[str]"             = None,  # path to annotations.txt for bbox dims
    gt_track_ids:   "Optional[list]"            = None,  # track ids per agent in B
) -> None:
    """
    Render observed history + predicted future trajectories over the source
    video and write an MP4.

    Parameters
    ----------
    mp4_path      : path to the source MP4
    pred_abs      : predicted positions in absolute pixel coords
                    deterministic -> [B, horizon, 2]
                    stochastic    -> [K, B, horizon, 2]
    last_obs_pos  : last observed absolute pixel position [B, 2]
                    used as the anchor connecting history to prediction
    x_abs         : absolute pixel positions for the observation window
                    shape [obs_len, B, 2]  (x[:, :, :2] BEFORE to_displacements)
    start_frame   : annotation frame index of the first observation frame
    output_path   : where to write the output MP4
    mode          : "deterministic" or "stochastic"
    det_pred_abs  : deterministic prediction [B, horizon, 2]; required when
                    mode="stochastic" to select the best sample per agent
    frameskip     : frameskip used during data loading (default 1)
    fps           : output FPS; defaults to source video FPS
    line_thickness: polyline thickness in pixels
    dot_radius    : trailing dot radius in pixels
    """
    # ------------------------------------------------------------------
    # Input validation & numpy conversion
    # ------------------------------------------------------------------
    def _to_np(t):
        if t is None:
            return None
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t, dtype=np.float32)

    pred_abs     = _to_np(pred_abs).astype(np.float32)
    last_obs_pos = _to_np(last_obs_pos).astype(np.float32)
    x_abs        = _to_np(x_abs).astype(np.float32)   # [obs_len, B, 2]
    det_pred_abs = _to_np(det_pred_abs)

    if mode not in ("deterministic", "stochastic"):
        raise ValueError(f"mode must be 'deterministic' or 'stochastic', got '{mode}'")

    if mode == "stochastic":
        if pred_abs.ndim != 4:
            raise ValueError(
                f"stochastic mode expects pred_abs [K, B, H, 2], got shape {pred_abs.shape}"
            )
        if det_pred_abs is None:
            raise ValueError("det_pred_abs is required when mode='stochastic'")
        det_pred_abs = det_pred_abs.astype(np.float32)
        pred_abs = _select_best_samples(pred_abs, det_pred_abs)  # [B, H, 2]
    else:
        if pred_abs.ndim != 3:
            raise ValueError(
                f"deterministic mode expects pred_abs [B, H, 2], got shape {pred_abs.shape}"
            )

    B, horizon, _ = pred_abs.shape
    obs_len       = x_abs.shape[0]

    # ------------------------------------------------------------------
    # Open video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {mp4_path}")

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps   = fps if fps is not None else src_fps
    total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_vid     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Frame indices we need: obs window + pred window
    # Each annotation step corresponds to `frameskip` video frames;
    # start_frame is the annotation index of the first obs frame.
    obs_frame_indices  = [start_frame + t * frameskip for t in range(obs_len)]
    pred_frame_indices = [obs_frame_indices[-1] + t * frameskip for t in range(1, horizon + 1)]
    all_frame_indices  = obs_frame_indices + pred_frame_indices

    first_vid_frame = min(all_frame_indices)
    last_vid_frame  = min(max(all_frame_indices), total_vid - 1)

    # ------------------------------------------------------------------
    # Writer + frames output dir
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (W, H_vid))
    frames_dir = os.path.splitext(output_path)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Coordinate transform: annotation space -> video pixel space
    #
    # SDD annotations are in reference.jpg space (1630x1948) with axes
    # rotated relative to the video. The correct transform is:
    #   1. scale:  sx = raw_x * W / ANN_W,  sy = raw_y * H_vid / ANN_H
    #   2. swap+flip: video_x = sy,  video_y = H_vid - sx
    # ------------------------------------------------------------------
    ANN_W, ANN_H = 1630, 1948

    def _to_video_coords(pts):
        """pts: [..., 2] in annotation space -> [..., 2] in video pixel space."""
        sx = pts[..., 0] * W     / ANN_W
        sy = pts[..., 1] * H_vid / ANN_H
        return np.stack([sx, sy], axis=-1)

    x_abs    = _to_video_coords(x_abs)
    pred_abs = _to_video_coords(pred_abs)

    # ------------------------------------------------------------------
    # Build per-annotation-step position lookup
    # annotation_step -> [B, 2] absolute positions
    # ------------------------------------------------------------------
    # obs steps: read from x_abs
    step_positions = {}  # annotation_step (0-indexed relative to start_frame) -> [B, 2]
    for t in range(obs_len):
        step_positions[t] = x_abs[t]  # [B, 2]

    # pred steps: from pred_abs
    for t in range(horizon):
        step_positions[obs_len + t] = pred_abs[:, t, :]   # [B, 2]

    total_steps = obs_len + horizon

    # ground truth positions for all steps (obs + pred)
    gt_positions = {}
    if gt_abs is not None:
        gt_abs_np = gt_abs if isinstance(gt_abs, np.ndarray) else np.array(gt_abs)
        gt_abs_transformed = _to_video_coords(gt_abs_np)  # apply sx,sy scale
        for t in range(gt_abs_transformed.shape[0]):
            gt_positions[t] = gt_abs_transformed[t]  # [B, 2]

    # load annotation bboxes if provided: {track_id: {frame: (xmin,ymin,xmax,ymax)}}
    ann_bboxes = {}
    if annot_path is not None and os.path.exists(annot_path):
        with open(annot_path) as _f:
            for _row in _f:
                _item = _row.strip().split()
                if not _item: continue
                _tid   = int(_item[0])
                _xmin  = float(_item[1]) * W     / ANN_W
                _ymin  = float(_item[2]) * H_vid / ANN_H
                _xmax  = float(_item[3]) * W     / ANN_W
                _ymax  = float(_item[4]) * H_vid / ANN_H
                _frame = int(_item[5])
                if _tid not in ann_bboxes:
                    ann_bboxes[_tid] = {}
                ann_bboxes[_tid][_frame] = (_xmin, _ymin, _xmax, _ymax)

    # map video frame index -> annotation step index (for the ones we annotate)
    frame_to_step = {}
    for step, vid_frame in enumerate(all_frame_indices):
        frame_to_step[vid_frame] = step

    # ------------------------------------------------------------------
    # Seek and render
    # ------------------------------------------------------------------
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_vid_frame)
    current_vid_frame = first_vid_frame

    while current_vid_frame <= last_vid_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if current_vid_frame in frame_to_step:
            step = frame_to_step[current_vid_frame]
            is_pred_step = step >= obs_len

            for agent_idx in range(B):
                color = _agent_color(agent_idx)

                # --- draw full observed history up to this step ---
                if not is_pred_step:
                    # history up to current step
                    hist_pts = np.stack(
                        [step_positions[s][agent_idx] for s in range(step + 1)],
                        axis=0,
                    )
                    _draw_trajectory(frame, hist_pts, color, line_thickness)
                    _draw_dot(frame, step_positions[step][agent_idx], color, dot_radius)

                # --- draw full observed history + predicted future ---
                else:
                    # full obs history (solid)
                    hist_pts = np.stack(
                        [step_positions[s][agent_idx] for s in range(obs_len)],
                        axis=0,
                    )
                    _draw_trajectory(frame, hist_pts, color, line_thickness)

                    # predicted up to current pred step (dashed appearance via
                    # shorter segments drawn at half-alpha)
                    pred_steps_so_far = step - obs_len + 1
                    # include last obs pos as anchor for the prediction line
                    pred_pts = np.stack(
                        [step_positions[obs_len - 1][agent_idx]]
                        + [step_positions[obs_len + s][agent_idx] for s in range(pred_steps_so_far)],
                        axis=0,
                    )
                    _draw_trajectory(frame, pred_pts, color, line_thickness + 1, alpha=0.75)
                    _draw_dot(frame, step_positions[step][agent_idx], color, dot_radius)

        # Draw ground truth red box for each agent
        if current_vid_frame in frame_to_step and gt_positions:
            step = frame_to_step[current_vid_frame]
            for agent_idx in range(B):
                if step not in gt_positions:
                    continue
                gt_pos = gt_positions[step][agent_idx]
                if not (np.isfinite(gt_pos).all() and (gt_pos < 1e8).all()):
                    continue
                gx, gy = int(gt_pos[0]), int(gt_pos[1])
                # use annotation bbox if available, else fixed size
                box_w, box_h = 30, 50
                if gt_track_ids is not None and ann_bboxes:
                    tid = gt_track_ids[agent_idx] if agent_idx < len(gt_track_ids) else None
                    ann_frame = start_frame + step
                    if tid is not None and tid in ann_bboxes and ann_frame in ann_bboxes[tid]:
                        _xmin, _ymin, _xmax, _ymax = ann_bboxes[tid][ann_frame]
                        box_w = int(_xmax - _xmin)
                        box_h = int(_ymax - _ymin)
                x0_b = max(0, gx - box_w//2)
                y0_b = max(0, gy - box_h//2)
                x1_b = min(frame.shape[1]-1, gx + box_w//2)
                y1_b = min(frame.shape[0]-1, gy + box_h//2)
                cv2.rectangle(frame, (x0_b, y0_b), (x1_b, y1_b), (0, 0, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        # save annotated frame as PNG only on annotation steps
        if current_vid_frame in frame_to_step:
            frame_path = os.path.join(frames_dir, f"frame_{current_vid_frame:06d}.png")
            cv2.imwrite(frame_path, frame)
        current_vid_frame += 1

    cap.release()
    writer.release()
    print(f"[visualize] Written {last_vid_frame - first_vid_frame + 1} frames -> {output_path}")


# ---------------------------------------------------------------------------
# Convenience wrapper that matches the eval loop in main.py
# ---------------------------------------------------------------------------

def visualize_from_eval_batch(
    mp4_path:       str,
    x:              "torch.Tensor",
    y_det_disp:     "torch.Tensor",
    last_obs:       "torch.Tensor",
    start_frame:    int,
    output_path:    str,
    mode:           str               = "deterministic",
    y_samp_disp:    "Optional[torch.Tensor]" = None,
    frameskip:      int               = 1,
    fps:            Optional[float]   = None,
    y_abs:          "Optional[torch.Tensor]" = None,   # [B, horizon, 2] GT future absolute
    annot_path:     "Optional[str]"          = None,   # path to annotations.txt
    gt_track_ids:   "Optional[list]"         = None,   # track ids per agent
) -> None:
    """
    Drop-in wrapper for the eval loop in main.py.

    Converts displacement predictions back to absolute pixel coords using
    to_absolute(), then calls visualize_predictions().

    `x` here is the RAW absolute-coordinate tensor from the dataloader
    (before to_displacements() was applied in main.py).

    start_frame should be item[5] from the dataloader batch — the annotation
    frame index stored as start_annotation_tid in data.py.
    """
    from main import to_absolute  # import here to avoid circular deps

    def _np(t):
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    x_abs_np    = _np(x[:, :, :2])      # [obs_len, B, 2]
    last_obs_np = _np(last_obs)          # [B, 2]

    # deterministic absolute
    y_det_abs = to_absolute(y_det_disp.unsqueeze(0), last_obs).squeeze(0)  # [B, H, 2]

    if mode == "stochastic":
        if y_samp_disp is None:
            raise ValueError("y_samp_disp required for stochastic mode")
        y_samp_abs = to_absolute(y_samp_disp, last_obs)   # [K, B, H, 2]
        # build full GT sequence: obs + pred
        _gt = None
        if y_abs is not None:
            import numpy as _np2
            _gt = _np2.concatenate([x_abs_np, _np(y_abs).transpose(1,0,2) if _np(y_abs).ndim==3 else _np(y_abs)[:,None,:]], axis=0) if False else _np2.concatenate([x_abs_np, _np(y_abs).swapaxes(0,1)], axis=0)
        visualize_predictions(
            mp4_path      = mp4_path,
            pred_abs      = y_samp_abs,
            last_obs_pos  = last_obs_np,
            x_abs         = x_abs_np,
            start_frame   = start_frame,
            output_path   = output_path,
            mode          = "stochastic",
            det_pred_abs  = y_det_abs,
            frameskip     = frameskip,
            fps           = fps,
            gt_abs        = _gt,
            annot_path    = annot_path,
            gt_track_ids  = gt_track_ids,
        )
    else:
        _gt = None
        if y_abs is not None:
            import numpy as _np2
            _gt = _np2.concatenate([x_abs_np, _np(y_abs).swapaxes(0,1)], axis=0)
        visualize_predictions(
            mp4_path      = mp4_path,
            pred_abs      = y_det_abs,
            last_obs_pos  = last_obs_np,
            x_abs         = x_abs_np,
            start_frame   = start_frame,
            output_path   = output_path,
            mode          = "deterministic",
            frameskip     = frameskip,
            fps           = fps,
            gt_abs        = _gt,
            annot_path    = annot_path,
            gt_track_ids  = gt_track_ids,
        )