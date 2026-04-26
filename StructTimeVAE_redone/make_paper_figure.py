"""
make_paper_figure.py

Generates a side-by-side paper figure from annotated frames.
Crops 300x300 around the agent, draws the full trajectory so far
and a purple bounding box at the current predicted position.

Usage:
    python3 make_paper_figure.py

Edit the CONFIG section below to match your paths and agent.
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------
FRAMES_DIR   = "/p3/home/mekeys/StructTimeVAE_redone/ckpt_full_v12/vis/epoch_0101/deathCircle_video0_batch0000_frames"
ANNOT_FILE   = "/p3/home/mekeys/SDD_embeddings/test/deathCircle_video0/annotations.txt"
OUTPUT_PATH  = "/p3/home/mekeys/StructTimeVAE_redone/ckpt_full_v12/vis/epoch_0101/paper_figure.png"

# Video frames to show side by side (must exist in FRAMES_DIR)
SHOW_FRAMES  = [0, 60, 144]

# Agent track id to visualize (the pedestrian we've been tracking)
TRACK_ID     = None   # None = auto-detect from first moving pedestrian

# Crop window half-size (300x300 total = 150 each side)
HALF         = 150

# Annotation -> video coordinate transform constants
ANN_W, ANN_H = 1630, 1948
VID_W, VID_H = 904, 1080

# Colors (BGR)
COLOR_OBS_TRAJ  = (0, 200, 255)    # amber — observed trajectory line
COLOR_PRED_TRAJ = (147, 20, 255)   # purple — predicted trajectory line
COLOR_PRED_BOX  = (147, 20, 255)   # purple — predicted bbox
COLOR_DOT       = (0, 200, 255)    # amber — current position dot
TRAJ_THICKNESS  = 2
BOX_THICKNESS   = 2
DOT_RADIUS      = 5

# Label below each panel
LABELS = [f"t = {f}" for f in SHOW_FRAMES]
# ---------------------------------------------------------------------------


def ann_to_video(rx, ry):
    """Scale annotation coords to video pixel coords."""
    return rx * VID_W / ANN_W, ry * VID_H / ANN_H


def load_track(annot_file, track_id=None):
    """
    Returns dict: {frame: (vid_x, vid_y)} for the chosen track.
    If track_id is None, picks the first moving pedestrian.
    """
    raw = {}   # track_id -> [(frame, x, y)]
    with open(annot_file) as f:
        for row in f:
            item = row.strip().split()
            if not item:
                continue
            tid   = int(item[0])
            label = item[9].replace('"', '').replace("'", '').strip().lower()
            if label != "pedestrian":
                continue
            x = (float(item[1]) + float(item[3])) / 2
            y = (float(item[2]) + float(item[4])) / 2
            frame = int(item[5])
            if tid not in raw:
                raw[tid] = []
            raw[tid].append((frame, x, y))

    if track_id is not None:
        pts = sorted(raw[track_id])
        return {f: ann_to_video(x, y) for f, x, y in pts}

    # auto: pick first pedestrian visible at the earliest SHOW_FRAMES[0]
    # that moves more than 20px over its trajectory
    target_frame = SHOW_FRAMES[0]
    for tid, pts in sorted(raw.items()):
        pts.sort()
        frames = [p[0] for p in pts]
        if target_frame not in frames:
            continue
        if len(pts) < 8:
            continue
        dx = pts[-1][1] - pts[0][1]
        dy = pts[-1][2] - pts[0][2]
        if (dx**2 + dy**2)**0.5 > 20:
            print(f"[figure] Auto-selected track {tid}")
            return {f: ann_to_video(x, y) for f, x, y in pts}

    raise RuntimeError("No suitable moving pedestrian track found.")


def crop_centered(img, cx, cy, half):
    """Crop half*2 x half*2 window centered on (cx,cy), clamped to image bounds."""
    H, W = img.shape[:2]
    x0 = max(0, int(cx) - half)
    y0 = max(0, int(cy) - half)
    x1 = min(W, x0 + half * 2)
    y1 = min(H, y0 + half * 2)
    # adjust origin if clamped at right/bottom edge
    x0 = x1 - half * 2
    y0 = y1 - half * 2
    x0 = max(0, x0)
    y0 = max(0, y0)
    return img[y0:y1, x0:x1], x0, y0


def draw_trajectory_on_crop(crop, track, all_frames_sorted, current_frame,
                             obs_end_frame, crop_ox, crop_oy):
    """
    Draw trajectory history up to current_frame on the cropped image.
    Positions are translated by crop origin (crop_ox, crop_oy).
    obs_end_frame: last frame of the observation window.
    """
    frames_up_to_now = [f for f in all_frames_sorted if f <= current_frame]

    if len(frames_up_to_now) < 2:
        return

    for i in range(1, len(frames_up_to_now)):
        f0, f1 = frames_up_to_now[i-1], frames_up_to_now[i]
        x0, y0 = track[f0]
        x1, y1 = track[f1]
        px0 = int(x0) - crop_ox
        py0 = int(y0) - crop_oy
        px1 = int(x1) - crop_ox
        py1 = int(y1) - crop_oy

        # obs segment = amber, pred segment = purple
        is_pred = f0 >= obs_end_frame
        color = COLOR_PRED_TRAJ if is_pred else COLOR_OBS_TRAJ
        cv2.line(crop, (px0, py0), (px1, py1), color, TRAJ_THICKNESS, cv2.LINE_AA)

    # dot at current position
    cx, cy = track[current_frame]
    px, py = int(cx) - crop_ox, int(cy) - crop_oy
    cv2.circle(crop, (px, py), DOT_RADIUS, COLOR_DOT, -1, cv2.LINE_AA)
    cv2.circle(crop, (px, py), DOT_RADIUS, (255, 255, 255), 1, cv2.LINE_AA)


def draw_bbox(crop, track, current_frame, crop_ox, crop_oy,
              annot_file, track_id_for_bbox):
    """
    Draw a bounding box at the agent's current position using the
    original annotation bbox size, transformed to video coords.
    """
    # Read the raw bbox for this track at this frame
    with open(annot_file) as f:
        for row in f:
            item = row.strip().split()
            if not item:
                continue
            if int(item[0]) != track_id_for_bbox:
                continue
            if int(item[5]) != current_frame:
                continue
            xmin = float(item[1]) * VID_W / ANN_W
            ymin = float(item[2]) * VID_H / ANN_H
            xmax = float(item[3]) * VID_W / ANN_W
            ymax = float(item[4]) * VID_H / ANN_H
            px0 = int(xmin) - crop_ox
            py0 = int(ymin) - crop_oy
            px1 = int(xmax) - crop_ox
            py1 = int(ymax) - crop_oy
            cv2.rectangle(crop, (px0, py0), (px1, py1),
                          COLOR_PRED_BOX, BOX_THICKNESS, cv2.LINE_AA)
            return


def add_label(img, text, font_scale=0.6, thickness=1):
    """Add a text label centered below the image."""
    H, W = img.shape[:2]
    label_h = 30
    canvas = np.zeros((H + label_h, W, 3), dtype=np.uint8)
    canvas[:H] = img
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (W - tw) // 2
    ty = H + (label_h + th) // 2
    cv2.putText(canvas, text, (tx, ty), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def main():
    # Load track
    track = load_track(ANNOT_FILE, TRACK_ID)
    all_frames_sorted = sorted(track.keys())

    # Determine track_id for bbox lookup
    raw_track_id = None
    with open(ANNOT_FILE) as f:
        for row in f:
            item = row.strip().split()
            if not item:
                continue
            label = item[9].replace('"', '').replace("'", '').strip().lower()
            if label != "pedestrian":
                continue
            frame = int(item[5])
            x = (float(item[1]) + float(item[3])) / 2
            y = (float(item[2]) + float(item[4])) / 2
            vx, vy = ann_to_video(x, y)
            # match by position at first show frame
            if frame == SHOW_FRAMES[0]:
                first_vx, first_vy = track[SHOW_FRAMES[0]]
                if abs(vx - first_vx) < 1 and abs(vy - first_vy) < 1:
                    raw_track_id = int(item[0])
                    print(f"[figure] Track id for bbox: {raw_track_id}")
                    break

    # Observation window end = last obs frame
    # Assume obs_len=8, frameskip=1: obs frames are first 8 in track
    obs_end_frame = all_frames_sorted[7] if len(all_frames_sorted) >= 8 else all_frames_sorted[-1]

    panels = []
    for i, vid_frame in enumerate(SHOW_FRAMES):
        frame_path = os.path.join(FRAMES_DIR, f"frame_{vid_frame:06d}.png")
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        img = cv2.imread(frame_path)

        # Find nearest track frame to this video frame
        nearest_frame = min(all_frames_sorted, key=lambda f: abs(f - vid_frame))
        cx, cy = track[nearest_frame]

        # Crop around agent
        crop, ox, oy = crop_centered(img, cx, cy, HALF)

        # Draw trajectory history up to this frame
        draw_trajectory_on_crop(crop, track, all_frames_sorted,
                                 nearest_frame, obs_end_frame, ox, oy)

        # Draw bbox only on predicted frames
        if nearest_frame > obs_end_frame and raw_track_id is not None:
            draw_bbox(crop, track, nearest_frame, ox, oy,
                      ANNOT_FILE, raw_track_id)

        # Add label
        crop = add_label(crop, LABELS[i])
        panels.append(crop)

    # Stack side by side with a small gap
    gap = np.zeros((panels[0].shape[0], 10, 3), dtype=np.uint8)
    figure = panels[0]
    for p in panels[1:]:
        figure = np.hstack([figure, gap, p])

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, figure)
    print(f"[figure] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()