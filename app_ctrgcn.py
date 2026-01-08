from matplotlib import pyplot as plt
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import mediapipe as mp
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import queue
from gtts import gTTS
from deep_translator import GoogleTranslator
import tempfile
import time
import os
import pandas as pd
import random
from contextlib import contextmanager
import sys
from collections import deque

# ==========================================
# 1. SETUP & UTILS
# ==========================================

POSE_LM = mp.solutions.holistic.PoseLandmark
HAND_LM = mp.solutions.holistic.HandLandmark

POSE_JOINTS = [
    ("torso", None), 
    ("left_shoulder", POSE_LM.LEFT_SHOULDER),
    ("right_shoulder", POSE_LM.RIGHT_SHOULDER),
    ("left_elbow", POSE_LM.LEFT_ELBOW),
    ("right_elbow", POSE_LM.RIGHT_ELBOW),
    ("left_wrist", POSE_LM.LEFT_WRIST),
    ("right_wrist", POSE_LM.RIGHT_WRIST),
]

HAND_NAMES = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

LEFT_HAND_JOINTS = [(n, lm) for n, lm in zip(HAND_NAMES, HAND_LM)]
RIGHT_HAND_JOINTS = [(n, lm) for n, lm in zip(HAND_NAMES, HAND_LM)]

JOINT_LIST = (
    [("pose", name, idx) for name, idx in POSE_JOINTS]
    + [("left_hand", name, idx) for name, idx in LEFT_HAND_JOINTS]
    + [("right_hand", name, idx) for name, idx in RIGHT_HAND_JOINTS]
)

JOINT_NAMES = [f"{src}:{name}" for src, name, _ in JOINT_LIST]
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}

@contextmanager
def suppress_cpp_logs():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    devnull = open(os.devnull, "w")
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        devnull.close()

class HolisticExtractor:
    def __init__(self, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        with suppress_cpp_logs():
            self.holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=int(model_complexity),
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def _process_frame(self, frame):
        return self.holistic.process(frame)

    def __call__(self, frames):
        t_len = len(frames)
        j_len = len(JOINT_LIST)
        skeleton = np.full((t_len, j_len, 3), np.nan, dtype=np.float32)
        
        torso_idx = JOINT_INDEX.get("pose:torso")
        l_shoulder_idx = JOINT_INDEX.get("pose:left_shoulder")
        r_shoulder_idx = JOINT_INDEX.get("pose:right_shoulder")

        for t, frame in enumerate(frames):
            results = self._process_frame(frame)
            if results is None: continue
            try:
                if results.pose_landmarks:
                    for local_idx, (_, idx_enum) in enumerate(POSE_JOINTS):
                        if idx_enum is None: continue
                        lm = results.pose_landmarks.landmark[idx_enum]
                        skeleton[t, local_idx] = [lm.x, lm.y, lm.z]
                    
                    if torso_idx is not None and l_shoulder_idx is not None and r_shoulder_idx is not None:
                        l_sh = skeleton[t, l_shoulder_idx]
                        r_sh = skeleton[t, r_shoulder_idx]
                        if not (np.isnan(l_sh).any() or np.isnan(r_sh).any()):
                            skeleton[t, torso_idx] = 0.5 * (l_sh + r_sh)
                        elif not np.isnan(l_sh).any(): skeleton[t, torso_idx] = l_sh
                        elif not np.isnan(r_sh).any(): skeleton[t, torso_idx] = r_sh
                
                if results.left_hand_landmarks:
                    offset = len(POSE_JOINTS)
                    for i, (name, idx_enum) in enumerate(LEFT_HAND_JOINTS):
                        lm = results.left_hand_landmarks.landmark[idx_enum]
                        skeleton[t, offset + i] = [lm.x, lm.y, lm.z]

                if results.right_hand_landmarks:
                    offset = len(POSE_JOINTS) + len(LEFT_HAND_JOINTS)
                    for i, (name, idx_enum) in enumerate(RIGHT_HAND_JOINTS):
                        lm = results.right_hand_landmarks.landmark[idx_enum]
                        skeleton[t, offset + i] = [lm.x, lm.y, lm.z]
            except Exception as e:
                print(f"Error frame {t}: {e}")
                continue
        return skeleton

def sample_frames_uniform(frames, target_len=80):
    n = len(frames)
    if target_len <= 0 or n == 0: return []
    if n < target_len:
        indices = np.linspace(0, n, num=target_len, endpoint=False).astype(np.int64)
        indices = np.clip(indices, 0, n - 1)
    else:
        indices = np.linspace(0, n - 1, num=target_len).astype(np.int64)
        indices[-1] = n - 1
    return [frames[int(i)] for i in indices]

def _interpolate_missing(arr):
    t, j, c = arr.shape
    out = arr.copy()
    for jj in range(j):
        for cc in range(c):
            series = out[:, jj, cc]
            valid = ~np.isnan(series)
            if not valid.any():
                out[:, jj, cc] = 0.0
                continue
            valid_idx = np.where(valid)[0]
            valid_vals = series[valid]
            interp = np.interp(np.arange(t), valid_idx, valid_vals)
            out[:, jj, cc] = interp
    return out

def _smooth_temporal(arr, window=3):
    if window <= 1: return arr
    pad = window // 2
    padded = np.pad(arr, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.zeros_like(arr)
    for jj in range(arr.shape[1]):
        for cc in range(arr.shape[2]):
            smoothed[:, jj, cc] = np.convolve(padded[:, jj, cc], kernel, mode="valid")
    return smoothed

def normalize_dual_stream(skeleton):
    t, j, c = skeleton.shape
    morph = skeleton.copy()
    traj = skeleton.copy()

    l_shoulder = JOINT_INDEX.get("pose:left_shoulder")
    r_shoulder = JOINT_INDEX.get("pose:right_shoulder")
    torso = JOINT_INDEX.get("pose:torso")
    r_wrist_pose = JOINT_INDEX.get("pose:right_wrist")
    l_wrist_pose = JOINT_INDEX.get("pose:left_wrist")
    l_hand_wrist = JOINT_INDEX.get("left_hand:wrist")
    r_hand_wrist = JOINT_INDEX.get("right_hand:wrist")
    l_middle_tip = JOINT_INDEX.get("left_hand:middle_tip")
    r_middle_tip = JOINT_INDEX.get("right_hand:middle_tip")

    for idx in range(t):
        frame = skeleton[idx]
        
        torso_node = frame[torso] if torso is not None else np.array([np.nan]*3)
        l_sh = frame[l_shoulder] if l_shoulder is not None else np.array([np.nan]*3)
        r_sh = frame[r_shoulder] if r_shoulder is not None else np.array([np.nan]*3)
        
        if not (np.isnan(l_sh).any() or np.isnan(r_sh).any()):
            shoulder_width = np.linalg.norm(l_sh - r_sh) + 1e-6
        else: shoulder_width = 1.0

        if not np.isnan(torso_node).any(): torso_origin = torso_node
        elif not (np.isnan(l_sh).any() or np.isnan(r_sh).any()): torso_origin = 0.5 * (l_sh + r_sh)
        elif not np.isnan(l_sh).any(): torso_origin = l_sh
        elif not np.isnan(r_sh).any(): torso_origin = r_sh
        else: torso_origin = np.zeros(3, dtype=np.float32)

        traj[idx] = (frame - torso_origin) / shoulder_width

        r_wrist = frame[r_hand_wrist] if r_hand_wrist is not None else np.array([np.nan]*3)
        l_wrist = frame[l_hand_wrist] if l_hand_wrist is not None else np.array([np.nan]*3)
        
        origin = r_wrist
        if np.isnan(origin).any(): origin = l_wrist
        if np.isnan(origin).any(): origin = frame[r_wrist_pose] if r_wrist_pose is not None else origin
        if np.isnan(origin).any(): origin = frame[l_wrist_pose] if l_wrist_pose is not None else origin
        if np.isnan(origin).any(): origin = torso_origin

        hand_scale = 1.0
        if not np.isnan(r_wrist).any() and r_middle_tip is not None and not np.isnan(frame[r_middle_tip]).any():
            hand_scale = np.linalg.norm(frame[r_middle_tip] - r_wrist) + 1e-6
        elif not np.isnan(l_wrist).any() and l_middle_tip is not None and not np.isnan(frame[l_middle_tip]).any():
            hand_scale = np.linalg.norm(frame[l_middle_tip] - l_wrist) + 1e-6
            
        morph[idx] = (frame - origin) / hand_scale

    return morph, traj

# ==========================================
# 2. MODEL (Updated to your Custom CTR-GCN)
# ==========================================
from typing import Optional

class CTRGC(nn.Module):
    """
    Channel-wise Topology Refinement Graph Convolution (CTR-GC).
    """
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, inter_channels: Optional[int] = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(8, out_channels // 4)
        self.register_buffer("A", A)
        self.PA = nn.Parameter(torch.zeros_like(A))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.conv_g = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_phi = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_rel = nn.Conv2d(inter_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        g = self.conv_g(x)
        theta = self.conv_theta(x).mean(dim=2)
        phi = self.conv_phi(x).mean(dim=2)
        rel = torch.tanh(theta.unsqueeze(-1) - phi.unsqueeze(-2))
        rel = self.conv_rel(rel)
        
        # Ensure A is on the correct device/dtype
        A = self.A.to(rel.dtype).to(rel.device)
        A = A.view(1, 1, v, v) + self.PA.view(1, 1, v, v)
        A_dyn = A + self.alpha.view(1, 1, 1, 1) * rel
        
        out = torch.einsum("nctv,ncuv->nctu", g, A_dyn)
        return out


class CTRGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, stride: int = 1, kernel_size_t: int = 9, dropout: float = 0.5):
        super().__init__()
        padding_t = (kernel_size_t - 1) // 2
        self.gcn = CTRGC(in_channels, out_channels, A)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size_t, 1), padding=(padding_t, 0), stride=(stride, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout)
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gcn = self.gcn(x)
        x_gcn = self.bn1(x_gcn)
        x_gcn = F.relu(x_gcn, inplace=True)
        x_tcn = self.tcn(x_gcn)
        x_tcn = self.bn2(x_tcn)
        x_tcn = self.drop(x_tcn)
        res = self.residual(x)
        out = F.relu(x_tcn + res, inplace=True)
        return out


class CTRGCNBackbone(nn.Module):
    def __init__(self, A: np.ndarray, hidden_dim: int = 96, num_blocks: int = 4, dropout: float = 0.5):
        super().__init__()
        # Ensure A is a tensor
        if not torch.is_tensor(A):
            A_tensor = torch.tensor(A, dtype=torch.float32)
        else:
            A_tensor = A.clone().detach().float()
            
        channels = [hidden_dim * (2 ** i) for i in range(num_blocks)]
        layers = []
        in_c = 3
        for idx, ch in enumerate(channels):
            stride = 2 if idx > 0 else 1
            layers.append(CTRGCNBlock(in_c, ch, A_tensor, stride=stride, dropout=dropout))
            in_c = ch
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DualStreamCTRGCN(nn.Module):
    def __init__(self, num_classes: int, A: np.ndarray, hidden_dim: int = 96, num_blocks: int = 4, dropout: float = 0.5):
        super().__init__()
        self.morph_stream = CTRGCNBackbone(A, hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=dropout)
        self.traj_stream = CTRGCNBackbone(A, hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=dropout)
        self.morph_drop = nn.Dropout(dropout)
        self.traj_drop = nn.Dropout(dropout)
        out_dim = self.morph_stream.out_channels
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, morph: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        m = self.morph_stream(morph)
        t = self.traj_stream(traj)
        m = self.morph_drop(m)
        t = self.traj_drop(t)
        m_feat = m.mean(dim=[2, 3])
        t_feat = t.mean(dim=[2, 3])
        feat = torch.cat([m_feat, t_feat], dim=1)
        logits = self.classifier(feat)
        return logits
    
def build_adjacency():
    j = len(JOINT_NAMES)
    A = np.zeros((j, j), dtype=np.float32)
    def connect(a, b):
        if a in JOINT_INDEX and b in JOINT_INDEX:
            ia, ib = JOINT_INDEX[a], JOINT_INDEX[b]
            A[ia, ib] = A[ib, ia] = 1.0
    connect("pose:torso", "pose:left_shoulder"); connect("pose:torso", "pose:right_shoulder")
    connect("pose:left_shoulder", "pose:right_shoulder")
    connect("pose:left_shoulder", "pose:left_elbow"); connect("pose:left_elbow", "pose:left_wrist")
    connect("pose:right_shoulder", "pose:right_elbow"); connect("pose:right_elbow", "pose:right_wrist")
    connect("pose:left_wrist", "left_hand:wrist"); connect("pose:right_wrist", "right_hand:wrist")
    edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
    off_l, off_r = len(POSE_JOINTS), len(POSE_JOINTS) + len(LEFT_HAND_JOINTS)
    for u,v in edges: connect(JOINT_NAMES[off_l+u], JOINT_NAMES[off_l+v])
    for u,v in edges: connect(JOINT_NAMES[off_r+u], JOINT_NAMES[off_r+v])
    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-6)))
    return deg_inv_sqrt @ A @ deg_inv_sqrt

# ==========================================
# 3. OVERLAY UTILS (MOTION TRAILS)
# ==========================================

def overlay_skeleton_on_frames(frames, skeleton, fps=20):
    T, J, _ = skeleton.shape
    H, W, _ = frames[0].shape
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    fourcc = cv2.VideoWriter_fourcc(*'vp80') 
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (W, H))
    for t in range(min(T, len(frames))):
        frame_rgb = frames[t].copy() 
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        coords = skeleton[t]
        for idx, (x, y, z) in enumerate(coords):
            if np.isnan(x) or np.isnan(y): continue
            cv2.circle(frame_bgr, (int(x*W), int(y*H)), 3, (0, 255, 0), -1) 
        info_text = f"Frame: {t}/{T} | FPS: {fps}"
        cv2.putText(frame_bgr, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out.write(frame_bgr)
    out.release()
    return out_file.name

# --- MOTION TRAIL HEATMAP GENERATOR ---
def overlay_heatmap_skeleton(frames, skeleton, fps=20):
    T, J, _ = skeleton.shape
    H, W, _ = frames[0].shape
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    fourcc = cv2.VideoWriter_fourcc(*'vp80') 
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (W, H))

    # Indices for Wrists
    lh_idx = JOINT_INDEX.get("left_hand:wrist", JOINT_INDEX.get("pose:left_wrist"))
    rh_idx = JOINT_INDEX.get("right_hand:wrist", JOINT_INDEX.get("pose:right_wrist"))

    # History buffers for trails (Max 15 frames of history)
    left_trail = deque(maxlen=15)
    right_trail = deque(maxlen=15)

    # Pre-calculate velocity
    diff = skeleton[1:] - skeleton[:-1]
    velocity = np.linalg.norm(diff, axis=2)
    velocity = np.vstack([np.zeros((1, J)), velocity]) 
    
    # Normalize velocity
    max_vel = np.percentile(velocity[~np.isnan(velocity)], 95) + 1e-6 
    norm_vel = np.clip(velocity / max_vel, 0, 1)

    for t in range(min(T, len(frames))):
        frame_rgb = frames[t].copy()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Dim the background to make trails pop
        frame_bgr = cv2.addWeighted(frame_bgr, 0.4, np.zeros_like(frame_bgr), 0.6, 0)

        # 1. Update Trails
        for hand_idx, trail in [(lh_idx, left_trail), (rh_idx, right_trail)]:
            if hand_idx is not None:
                coords = skeleton[t, hand_idx]
                if not (np.isnan(coords[0]) or np.isnan(coords[1])):
                    vel = norm_vel[t, hand_idx]
                    if np.isnan(vel): vel = 0
                    cx, cy = int(coords[0]*W), int(coords[1]*H)
                    trail.append((cx, cy, vel))

        # 2. Draw Trails
        for trail in [left_trail, right_trail]:
            for i in range(1, len(trail)):
                pt1 = (trail[i-1][0], trail[i-1][1])
                pt2 = (trail[i][0], trail[i][1])
                intensity = trail[i][2]
                
                # Color Map: Blue (Slow) -> Red (Fast)
                color = (int(255 * (1 - intensity)), 50, int(255 * intensity))
                thickness = int(4 * (i / len(trail))) + 1
                
                cv2.line(frame_bgr, pt1, pt2, color, thickness)

        # 3. Draw Active Joints (Only Hands and Arms)
        for idx, (x, y, z) in enumerate(skeleton[t]):
            if np.isnan(x) or np.isnan(y): continue
            
            # Filter: Only draw if velocity is significant OR it's a wrist
            if norm_vel[t, idx] > 0.15 or idx in [lh_idx, rh_idx]:
                intensity = norm_vel[t, idx]
                if np.isnan(intensity): intensity = 0
                color = (int(255 * (1 - intensity)), 100, int(255 * intensity))
                radius = 3 if idx not in [lh_idx, rh_idx] else 6
                cv2.circle(frame_bgr, (int(x*W), int(y*H)), radius, color, -1) 
        
        cv2.putText(frame_bgr, "Motion Trace (Trail=Shape, Color=Speed)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        out.write(frame_bgr)
    out.release()
    return out_file.name

# --- MOTION INSIGHTS GENERATOR ---
def extract_motion_insights(skeleton):
    """Calculates kinematic stats from skeleton data."""
    # 1. Calculate Overall Energy (Mean Velocity)
    # shape: (T, J, 3)
    diff = skeleton[1:] - skeleton[:-1]
    velocity = np.linalg.norm(diff, axis=2) # (T-1, J)
    
    # Filter out stationary points (noise) and get mean velocity of active joints
    active_vel = velocity[velocity > 0.005] # Threshold to ignore jitter
    if len(active_vel) > 0:
        avg_energy = np.mean(active_vel) * 100 # Scale for readability
    else:
        avg_energy = 0.0

    # 2. Generate Text Feedback
    insights = {}
    
    # Energy Feedback
    insights['energy_score'] = avg_energy
    if avg_energy < 1.0:
        insights['energy_text'] = "🐢 Slow / Subtle"
        insights['energy_tip'] = "Your movement is very small. Try signing with more energy!"
    elif avg_energy > 4.0:
        insights['energy_text'] = "⚡ Fast / Dynamic"
        insights['energy_tip'] = "Great energy! Make sure to maintain control."
    else:
        insights['energy_text'] = "✅ Balanced"
        insights['energy_tip'] = "Good signing speed."

    # 3. Size/Expansion (Bounding Box of Hands)
    # Get all hand coordinates
    valid_coords = skeleton[~np.isnan(skeleton).any(axis=2)]
    if len(valid_coords) > 0:
        # Calculate bounding box diagonal roughly
        min_x, max_x = np.min(valid_coords[:, 0]), np.max(valid_coords[:, 0])
        min_y, max_y = np.min(valid_coords[:, 1]), np.max(valid_coords[:, 1])
        area = (max_x - min_x) * (max_y - min_y)
        insights['size_score'] = area
        if area < 0.05:
            insights['size_text'] = "⚠️ Small"
            insights['size_tip'] = "Sign Larger: Use more space."
        else:
            insights['size_text'] = "✅ Optimal"
            insights['size_tip'] = "Good use of signing space."
    else:
        insights['size_score'] = 0
        insights['size_text'] = "❓ Not Detected"
        insights['size_tip'] = "No motion detected."
        
    return insights

def get_model_specs(model):
    """Calculates real model parameter counts (Option C)"""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# ==========================================
# 🧠 STEP 1: XAI ENGINES 
# ==========================================
def compute_dual_xai(model, morph_tensor, traj_tensor, target_class_idx):
    """Computes attribution for Dual-Stream ST-GCN."""
    model.eval()
    m_in = morph_tensor.detach().clone(); m_in.requires_grad = True
    t_in = traj_tensor.detach().clone(); t_in.requires_grad = True
    
    output = model(m_in, t_in)
    score = output[0, target_class_idx]
    
    model.zero_grad()
    score.backward()
    
    sal_m = torch.sqrt(torch.sum(m_in.grad.data.cpu() ** 2, dim=1))
    sal_t = torch.sqrt(torch.sum(t_in.grad.data.cpu() ** 2, dim=1))
    
    total_saliency = (sal_m + sal_t).squeeze(0)
    total_saliency = (total_saliency - total_saliency.min()) / (total_saliency.max() - total_saliency.min() + 1e-9)
    return total_saliency.numpy()

def render_xai_charts(saliency_matrix, joint_names):
    """Upgraded Level 1 & 2 Charts with readable names."""
    # Data Prep
    joint_scores = saliency_matrix.sum(axis=0)
    top_indices = joint_scores.argsort()[-5:][::-1]
    
    # USE NEW READABLE NAMES
    top_names = [get_readable_joint_name(joint_names[i]) for i in top_indices]
    
    temporal_scores = saliency_matrix.sum(axis=1)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Chart 1: Anatomy (Horizontal Bar)
    colors = plt.cm.viridis(np.linspace(0.4, 0.8, 5))
    ax1.barh(top_names, joint_scores[top_indices], color=colors)
    ax1.set_title("Most Important Body Parts", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Attention Score")
    ax1.invert_yaxis() # Highest at top
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Chart 2: Timing (Area Chart)
    ax2.plot(temporal_scores, color='#ff4b4b', lw=2)
    ax2.fill_between(range(len(temporal_scores)), temporal_scores, color='#ff4b4b', alpha=0.2)
    ax2.set_title("Attention Over Time", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Video Frame")
    ax2.set_ylabel("Intensity")
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_xai_video(raw_frames, saliency_matrix, skeleton_coords):
    # 1. Safety Check
    if not raw_frames or len(raw_frames) == 0:
        print("Error: No frames to process.")
        return None
        
    # 2. Setup Video Writer (USE WEBM for Streamlit Compatibility)
    height, width, _ = raw_frames[0].shape
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm') # Changed to .webm
    
    # 'vp80' is the safest codec for web browsers
    fourcc = cv2.VideoWriter_fourcc(*'vp80') 
    out = cv2.VideoWriter(temp_file.name, fourcc, 15.0, (width, height))
    
    limit = min(len(raw_frames), saliency_matrix.shape[0], len(skeleton_coords))
    
    for t in range(limit):
        # 3. Color Conversion (Streamlit uses RGB, OpenCV needs BGR)
        frame_rgb = raw_frames[t].copy()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # Fix Blue/Red tint
        
        # Dim background
        frame = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0)
        
        frame_importance = saliency_matrix[t]
        
        for v_idx, score in enumerate(frame_importance):
            try:
                # 4. Handle Coordinate Unpacking Safely
                coords = skeleton_coords[t][v_idx]
                if len(coords) == 3:
                    x, y, z = coords
                else:
                    x, y = coords
                    
                if np.isnan(x) or np.isnan(y): continue
                
                cx, cy = int(x * width), int(y * height)
                
                # Color: Blue (Low) -> Red (High)
                color = (int(255 * (1-score)), 50, int(255 * score))
                radius = int(3 + (10 * score))
                
                # Threshold to keep video clean
                if score > 0.05:
                    cv2.circle(frame, (cx, cy), radius, color, -1)
            except Exception:
                pass
        
        cv2.putText(frame, "AI Attention (Red=High)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        out.write(frame)
        
    out.release()
    return temp_file.name

# ==========================================
# 🧠 XAI INTERPRETABILITY HELPERS
# ==========================================

def get_readable_joint_name(raw_name):
    """Translates technical joint names to plain English."""
    name_map = {
        "pose:torso": "Core / Chest",
        "pose:left_shoulder": "Left Shoulder", "pose:right_shoulder": "Right Shoulder",
        "pose:left_elbow": "Left Elbow", "pose:right_elbow": "Right Elbow",
        "pose:left_wrist": "Left Wrist", "pose:right_wrist": "Right Wrist",
        "left_hand:wrist": "Left Hand Base", "right_hand:wrist": "Right Hand Base",
    }
    # Handle specific fingers simply
    if "thumb" in raw_name: return "Thumb"
    if "index" in raw_name: return "Index Finger"
    if "middle" in raw_name: return "Middle Finger"
    if "ring" in raw_name: return "Ring Finger"
    if "pinky" in raw_name: return "Pinky Finger"
    
    return name_map.get(raw_name, raw_name.split(":")[-1].replace("_", " ").title())

def generate_xai_narrative(saliency_matrix, joint_names):
    """
    Generates a granular text summary (Finger-level detail).
    """
    # 1. WHERE (Spatial) - granular grouping
    joint_scores = saliency_matrix.sum(axis=0)
    total_score = joint_scores.sum() + 1e-9
    
    # Initialize granular buckets
    # We use a dictionary to accumulate scores for specific parts
    detailed_parts = {}
    
    for i, name in enumerate(joint_names):
        score = joint_scores[i]
        
        # --- Logic to map technical names to Human Parts ---
        # 1. RIGHT HAND DETAILED
        if "right_hand" in name:
            if "thumb" in name: group = "Right Thumb"
            elif "index" in name: group = "Right Index Finger"
            elif "middle" in name: group = "Right Middle Finger"
            elif "ring" in name: group = "Right Ring Finger"
            elif "pinky" in name: group = "Right Pinky Finger"
            else: group = "Right Palm/Wrist" # wrist, cmc, etc.
            
        # 2. LEFT HAND DETAILED
        elif "left_hand" in name:
            if "thumb" in name: group = "Left Thumb"
            elif "index" in name: group = "Left Index Finger"
            elif "middle" in name: group = "Left Middle Finger"
            elif "ring" in name: group = "Left Ring Finger"
            elif "pinky" in name: group = "Left Pinky Finger"
            else: group = "Left Palm/Wrist"
            
        # 3. BODY & ARMS
        elif "elbow" in name: group = "Elbows"
        elif "shoulder" in name: group = "Shoulders"
        elif "torso" in name: group = "Torso/Core"
        else: group = "General Body"

        # Add to bucket
        detailed_parts[group] = detailed_parts.get(group, 0) + score
        
    # Find the winner
    dominant_part = max(detailed_parts, key=detailed_parts.get)
    region_pct = (detailed_parts[dominant_part] / total_score) * 100
    
    # 2. WHEN (Temporal)
    temporal_scores = saliency_matrix.sum(axis=1)
    peak_frame = np.argmax(temporal_scores)
    total_frames = len(temporal_scores)
    
    if peak_frame < total_frames * 0.33: timing = "start"
    elif peak_frame < total_frames * 0.66: timing = "middle"
    else: timing = "end"
    
    # 3. Construct Narrative
    # We add logic to make the sentence sound natural
    narrative = (
        f"The AI focused specifically on your **{dominant_part}** ({region_pct:.0f}% importance). "
        f"The crucial movement happened at the **{timing}** of the sign."
    )
    
    return narrative, peak_frame

def get_peak_frame_image(frames, peak_idx, saliency_matrix, skeleton):
    """Extracts the single most important frame and highlights the focus area."""
    if peak_idx >= len(frames): return None
    
    # Get the frame
    img = frames[peak_idx].copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
    h, w, _ = img.shape
    
    # Darken background significantly to make focus pop
    overlay = img.copy()
    img = cv2.addWeighted(img, 0.2, np.zeros_like(img), 0.8, 0) # Very dark background
    
    # Get attention for this frame
    frame_att = saliency_matrix[peak_idx]
    
    # Find the specific joint with max attention in this frame
    max_joint_idx = np.argmax(frame_att)
    
    # Draw ALL joints faintly
    for i, score in enumerate(frame_att):
        try:
            x, y, z = skeleton[peak_idx][i]
            if np.isnan(x) or np.isnan(y): continue
            cx, cy = int(x * w), int(y * h)
            
            # Color logic
            color = (0, 255, 255) # Yellow default
            radius = 2
            if score > 0.1: # Significant
                radius = int(5 + (20 * score))
                color = (0, 0, 255) # Red
                cv2.circle(img, (cx, cy), radius, color, -1)
                # Add "Glow"
                cv2.circle(img, (cx, cy), radius + 10, color, 2)
        except: pass
        
    return img

# ==========================================
# 4. VIDEO PROCESSOR 
# ==========================================

def load_video_frames_from_file(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

# ==========================================
# 🧠 STEP 2: UPDATE PIPELINE 
# ==========================================
def process_prediction_pipeline(raw_frames, model, class_names):
    start_time = time.time()
    if len(raw_frames) < 10: return None
        
    # 1. Preprocessing
    target_len = 80
    sampled_frames = sample_frames_uniform(raw_frames, target_len)
    extractor = HolisticExtractor(model_complexity=2) 
    skeleton_raw = extractor(sampled_frames)
    skeleton_interp = _interpolate_missing(skeleton_raw)
    skeleton_smooth = _smooth_temporal(skeleton_interp)
    morph, traj = normalize_dual_stream(skeleton_smooth)
    preprocess_end = time.time()
    
    # 2. Inference
    m_t = torch.tensor(morph).float().permute(2,0,1).unsqueeze(0)
    t_t = torch.tensor(traj).float().permute(2,0,1).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(m_t, t_t)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_idxs = torch.topk(probs, 3, dim=1)
    
    inference_end = time.time()
    
    # 3. Packaging Results
    top_probs = top_probs.squeeze().tolist()
    top_idxs = top_idxs.squeeze().tolist()
    top_3_preds = [(class_names[idx], prob) for prob, idx in zip(top_probs, top_idxs)]
    best_pred = top_3_preds[0][0]
    best_conf = top_3_preds[0][1]
    
    # 4. Generate Visuals
    overlay_path = overlay_skeleton_on_frames(sampled_frames, skeleton_smooth)
    heatmap_path = overlay_heatmap_skeleton(sampled_frames, skeleton_smooth)
    insights = extract_motion_insights(skeleton_smooth)
    
    perf_metrics = {
        "preprocess": preprocess_end - start_time,
        "inference": inference_end - preprocess_end,
        "input_frames": len(raw_frames)
    }

    # CRITICAL: Return Dictionary with XAI Data
    return {
        "pred": best_pred,
        "conf": best_conf,
        "overlay": overlay_path,
        "heatmap": heatmap_path,
        "top3": top_3_preds,
        "insights": insights,
        "metrics": perf_metrics,
        "xai_data": {
            "morph": m_t, "traj": t_t, 
            "skeleton": skeleton_smooth, "frames": sampled_frames
        }
    }

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.recording = False
        self.raw_frame_buffer = [] 
        self.queue = None 
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.recording:
            resized = cv2.resize(img, (640, 480))
            self.raw_frame_buffer.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            cv2.circle(img, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(img, f"REC {len(self.raw_frame_buffer)}", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Standby", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if len(self.raw_frame_buffer) > 0:
                if self.queue:
                    self.queue.put({"type": "process", "frames": self.raw_frame_buffer})
                self.raw_frame_buffer = [] 
        return img

# ==========================================
# 5. MAIN UI LOGIC
# ==========================================

st.set_page_config(page_title="MSL App Demo", layout="wide")

# -- Session States --
if "result_queue" not in st.session_state: st.session_state.result_queue = queue.Queue()

# Toast Control State
if "last_loaded_model_path" not in st.session_state: st.session_state.last_loaded_model_path = None

# Live Mode States
if "is_recording" not in st.session_state: st.session_state.is_recording = False
if "waiting_for_prediction" not in st.session_state: st.session_state.waiting_for_prediction = False
if "live_last_result" not in st.session_state: st.session_state.live_last_result = None

# Practice Mode States
if "practice_active" not in st.session_state: st.session_state.practice_active = False
if "practice_streak" not in st.session_state: st.session_state.practice_streak = 0
if "practice_target" not in st.session_state: st.session_state.practice_target = None
if "practice_result_shown" not in st.session_state: st.session_state.practice_result_shown = False
if "practice_is_recording" not in st.session_state: st.session_state.practice_is_recording = False
if "practice_waiting" not in st.session_state: st.session_state.practice_waiting = False
if "practice_last_data" not in st.session_state: st.session_state.practice_last_data = None 
if "practice_category" not in st.session_state: st.session_state.practice_category = "All" 

st.title("🤟 AI Malaysian Sign Language Interpreter")

# Config
CLASS_NAMES = sorted(['abang', 'ada', 'ambil', 'anak_lelaki', 'anak_perempuan', 'apa', 'apa_khabar', 'arah', 'assalamualaikum', 'baca', 'bagaimana', 'bahasa_isyarat', 'baik', 'bapa', 'bapa_saudara', 'bas', 'bawa', 'beli', 'beli_2', 'berapa', 'berjalan', 'berlari', 'bila', 'bola', 'boleh', 'bomba', 'buang', 'buat', 'curi', 'dapat', 'dari', 'emak', 'emak_saudara', 'hari', 'hi', 'hujan', 'jahat', 'jam', 'jangan', 'jumpa', 'kacau', 'kakak', 'keluarga', 'kereta', 'kesakitan', 'lelaki', 'lemak', 'lupa', 'main', 'makan', 'mana', 'marah', 'mari', 'masa', 'masalah', 'minum', 'mohon', 'mohon_2', 'nasi', 'nasi_lemak', 'panas', 'panas_2', 'pandai', 'pandai_2', 'payung', 'pen', 'pensil', 'perempuan', 'pergi', 'pergi_2', 'perlahan', 'perlahan_2', 'pinjam', 'polis', 'pukul', 'ribut', 'sampai', 'saudara', 'sejuk', 'sekolah', 'siapa', 'sudah', 'suka', 'tandas', 'tanya', 'teh_tarik', 'teksi', 'tidur', 'tolong'])

# --- LESSON MODULES CATEGORIES ---
CATEGORIES = {
    "All": CLASS_NAMES,
    "Family 👨‍👩‍👧": ['abang', 'anak_lelaki', 'anak_perempuan', 'bapa', 'bapa_saudara', 'emak', 'emak_saudara', 'kakak', 'keluarga', 'lelaki', 'perempuan', 'saudara'],
    "Greetings & Social 👋": ['assalamualaikum', 'hi', 'apa_khabar', 'baik', 'baik_2', 'jumpa', 'tolong', 'apa', 'bagaimana', 'berapa', 'bila', 'mana', 'siapa', 'tanya'],
    "Verbs & Actions 🏃": ['ambil', 'baca', 'bawa', 'beli', 'beli_2', 'berjalan', 'berlari', 'buang', 'buat', 'curi', 'dapat', 'kacau', 'main', 'makan', 'minum', 'mohon', 'pergi', 'pergi_2', 'pinjam', 'pukul', 'tidur'],
    "Food & Drink 🍔": ['nasi', 'nasi_lemak', 'teh_tarik', 'lemak'],
    "Places & Transport 🚌": ['bas', 'kereta', 'teksi', 'sekolah', 'tandas', 'bomba', 'polis'],
    "Objects ✏️": ['bola', 'payung', 'pen', 'pensil', 'jam'],
    "Nature & Weather 🌦️": ['hari', 'hujan', 'panas', 'panas_2', 'ribut', 'sejuk'],
    "Feelings & States 💭": ['jahat', 'kesakitan', 'marah', 'pandai', 'pandai_2', 'perlahan', 'perlahan_2', 'suka', 'ada', 'sudah', 'boleh', 'jangan']
}

@st.cache_resource
def get_model(path):
    """Loads the CTR-GCN model."""
    
    # Defaults from your code snippet: hidden_dim=96, num_blocks=4
    # If your teammate changed these during training, update them here!
    model = DualStreamCTRGCN(
        num_classes=len(CLASS_NAMES), 
        A=build_adjacency(), 
        hidden_dim=96, 
        num_blocks=4, 
        dropout=0.5
    )
    
    meta = {}
    try:
        # Load file
        checkpoint = torch.load(path, map_location='cpu')
        
        # Handle state dict structure
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
        
        # Load weights
        model.load_state_dict(state_dict)
        model.eval()

        # Extract metadata if available
        if isinstance(checkpoint, dict):
            for k, v in checkpoint.items():
                if k not in ['model', 'state_dict', 'optimizer', 'scheduler', 'model_state_dict']:
                    if isinstance(v, (int, float, str, bool)):
                        meta[k] = v
        
        return model, meta

    except Exception as e:
        print(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None, {}

# ==========================================
# SIDEBAR LAYOUT
# ==========================================
st.sidebar.title("⚙️ Settings")
target_lang = st.sidebar.selectbox("Translate results to:", ["en", "es", "fr", "de", "zh-CN", "ms"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("📚 Resources")
st.sidebar.markdown("Want to learn more signs?  \n[**Visit BIM Sign Bank**](https://www.bimsignbank.org)")

st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ Advanced Settings", expanded=False):
    model_path = st.text_input("Model Checkpoint Path", "checkpoints/best_model_ctrgcn_cb3.pth")
    if st.button("Reload Model"):
        st.cache_resource.clear()
        st.session_state.last_loaded_model_path = None
        st.rerun()
    
    st.markdown("---")
    dev_mode = st.checkbox("🛠️ Enable Developer View", value=False, key="dev_mode")

# Load Model
model, model_meta = get_model(model_path) # Unpack tuple
if model: 
    if st.session_state.last_loaded_model_path != model_path:
        st.toast("Model Loaded Successfully!", icon="✅")
        st.session_state.last_loaded_model_path = model_path
else: 
    st.sidebar.error("Model Not Found. Check path.")

# --- DYNAMIC MAIN TABS ---
main_tabs_list = ["📹 Live Recording", "📤 Upload Video", "🎮 Practice Mode"]
if dev_mode: main_tabs_list.append("🛠️ Model Health")

tabs = st.tabs(main_tabs_list)
tab_live, tab_upload, tab_practice = tabs[0], tabs[1], tabs[2]

# --- TAB 1: LIVE WEBCAM ---
with tab_live:
    with st.expander("ℹ️ Instructions: How to use Live Recording Mode", expanded=False):
        st.markdown("""
        1. **Start the Camera**: Click 'Start' in the video player below.
        2. **Record**: Click the **🔴 Start Recording** button.
        3. **Perform Sign**: Make sure your upper body and hands are visible.
        4. **Stop**: Click **⏹️ Stop & Process** to analyze.
        """)

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        btn_placeholder = st.empty()
        ctx = webrtc_streamer(
            key="raw-recorder",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"frameRate": {"ideal": 30}}, "audio": False},
            async_processing=True,
        )
        
        camera_active = ctx.state.playing if ctx.state else False

        with btn_placeholder:
            btn_label = "🔴 Start Recording" if not st.session_state.is_recording else "⏹️ Stop & Process"
            if st.button(btn_label, key="live_btn", disabled=not camera_active):
                if st.session_state.is_recording:
                    st.session_state.is_recording = False
                    st.session_state.waiting_for_prediction = True
                else:
                    st.session_state.is_recording = True
                    st.session_state.waiting_for_prediction = False
                    st.session_state.live_last_result = None
                st.rerun()
            if not camera_active:
                st.caption("⚠️ Please start the camera to record.")

        if ctx.video_transformer:
            ctx.video_transformer.recording = st.session_state.is_recording
            ctx.video_transformer.queue = st.session_state.result_queue

    with col2:
            st.subheader("Result Analysis")
            status_box = st.empty()
            if not st.session_state.is_recording and not st.session_state.waiting_for_prediction and st.session_state.live_last_result is None:
                st.info("Record a sign to see analysis.")
            # --- 1. HANDLE WAITING STATE ---
            if st.session_state.waiting_for_prediction:
                status_box.info("⏳ Processing video...")
                with st.spinner("Analyzing..."):
                    try:
                        msg = st.session_state.result_queue.get(timeout=5)
                        if msg['type'] == 'process':
                            raw_frames = msg['frames']
                            # Run Pipeline
                            result_dict = process_prediction_pipeline(raw_frames, model, CLASS_NAMES)
                            # Save to Session State
                            st.session_state.live_last_result = result_dict
                            
                            st.session_state.waiting_for_prediction = False
                            st.rerun() 
                    except queue.Empty:
                        status_box.error("Timeout. Try again.")
                        st.session_state.waiting_for_prediction = False

            # --- 2. DISPLAY RESULTS ---
            if st.session_state.live_last_result:
                res = st.session_state.live_last_result
                
                # Extract variables safely
                pred = res.get("pred")
                conf = res.get("conf")
                vid_path = res.get("overlay")
                heatmap_path = res.get("heatmap")
                top3 = res.get("top3", [])
                insights = res.get("insights", {})
                metrics = res.get("metrics", {})
                xai_data = res.get("xai_data", None)

                if pred:
                    status_box.success("Analysis Complete!")
                    
                    # MATCHED TABS: Analysis, AI Coaching, Deep XAI
                    live_tabs = st.tabs(["📊 Analysis", "🤖 AI Coaching", "🧠 Deep XAI"])
                    
                    # === TAB 1: ANALYSIS ===
                    with live_tabs[0]:
                        st.caption("Standard Structure")
                        if vid_path: st.video(vid_path)
                        
                        # Prediction Card
                        with st.container(border=True):
                             st.markdown(f"### 🔮 I think you signed: **{pred.replace('_', ' ').title()}**")

                        # Confidence Card
                        with st.container(border=True):
                            st.markdown("##### 🎯 AI Confidence Score")
                            if len(top3) >= 1:
                                score = top3[0][1]
                                st.progress(score)
                                if score > 0.9: st.caption("🌟 Excellent Match")
                                elif score > 0.7: st.caption("✅ Good Match")
                                else: st.caption("⚠️ Uncertain / Low Confidence")
                                st.metric("🥇 Top Match", top3[0][0], f"{score:.1%}")

                            if len(top3) >= 2:
                                st.metric("🥈 Alternative", top3[1][0], f"{top3[1][1]:.1%}", delta_color="off")
                            if len(top3) >= 3:
                                st.metric("🥉 Alternative", top3[2][0], f"{top3[2][1]:.1%}", delta_color="off")
                        
                        # Latency Metrics (Dev Mode)
                        if dev_mode and metrics: 
                             with st.expander("⚡ System Latency (Dev)", expanded=False):
                                k1, k2, k3 = st.columns(3)
                                k1.metric("Preproc", f"{metrics.get('preprocess',0)*1000:.0f}ms")
                                k2.metric("Inference", f"{metrics.get('inference',0)*1000:.0f}ms")
                                total_t = metrics.get('total_time', metrics.get('preprocess',0) + metrics.get('inference',0))
                                k3.metric("Total", f"{total_t:.2f}s")

                        st.divider()
                        
                        # Translation & Speech Section
                        try:
                            clean_pred = pred.replace("_", " ")
                            trans = GoogleTranslator(source='auto', target=target_lang).translate(clean_pred)
                            st.markdown("#### 🗣️ Translation & Speech")
                            t1, t2 = st.columns(2)
                            with t1: st.info(f"**Original (BM):**\n{clean_pred.title()}")
                            with t2: st.success(f"**Translated ({target_lang}):**\n{trans}")
                            
                            # Audio Generation
                            tts = gTTS(trans, lang=target_lang)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                                tts.save(fp.name)
                                st.audio(fp.name, format="audio/mp3")
                        except Exception as e:
                            # Silently fail or show small error if internet is down for Translator
                            st.caption(f"Translation unavailable: {e}")

                    # === TAB 2: AI COACHING (Matches Upload Tab) ===
                    with live_tabs[1]:
                        st.caption("Velocity Trails (Red = Fast)")
                        if heatmap_path: st.video(heatmap_path)
                        
                        if insights:
                            st.divider()
                            st.markdown("##### 🤖 AI Motion Coaching (Beta)")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.caption("Kinetic Energy")
                                st.markdown(f"**{insights.get('energy_text', '-')}**")
                                st.caption(insights.get("energy_tip", ""))
                            with c2:
                                st.caption("Space Usage")
                                st.markdown(f"**{insights.get('size_text', '-')}**")
                                st.caption(insights.get("size_tip", ""))

                    # === TAB 3: DEEP XAI ===
                with live_tabs[2]: 
                    conf = res["conf"]
                    if conf > 0.4 and "xai_data" in res:
                        xai_data = res["xai_data"]
                        
                        # 1. Compute Saliency
                        saliency = compute_dual_xai(model, xai_data["morph"], xai_data["traj"], 0)
                        
                        # 2. GENERATE NARRATIVE (The "Why")
                        narrative, peak_frame_idx = generate_xai_narrative(saliency, JOINT_NAMES)
                        
                        st.info(f"💡 **AI Insight:** {narrative}")
                        
                        # 3. SHOW THE "DECIDING MOMENT"
                        col_x1, col_x2 = st.columns([1, 1])
                        
                        with col_x1:
                            st.markdown("#### 📸 The Deciding Frame")
                            st.caption("The exact moment the AI made its decision.")
                            peak_img = get_peak_frame_image(xai_data["frames"], peak_frame_idx, saliency, xai_data["skeleton"])
                            if peak_img is not None:
                                st.image(peak_img, channels="BGR", use_container_width=True)
                        
                        with col_x2:
                            st.markdown("#### 🎥 Full Attention Replay")
                            st.caption("Red Glow = High Importance")
                            xai_vid = generate_xai_video(xai_data["frames"], saliency, xai_data["skeleton"])
                            if xai_vid: st.video(xai_vid)

                        # 4. SHOW IMPROVED CHARTS
                        st.divider()
                        st.markdown("#### 📊 Detailed Breakdown")
                        fig = render_xai_charts(saliency, JOINT_NAMES)
                        st.pyplot(fig)
                        
                    else:
                        st.warning("⚠️ Confidence too low for deep analysis (Need > 40%) or XAI data missing.")
                

# --- TAB 2: VIDEO UPLOAD ---
with tab_upload:
    with st.expander("ℹ️ Instructions: Uploading Videos", expanded=False):
        st.markdown("""
        1. **Prepare Video**: Use MP4, MOV, or AVI format.
        2. **Upload**: Drag and drop your file below.
        3. **Process**: Click 'Process Video' to analyze.
        """)

    st.markdown("Upload a video to interpret your sign.")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    
    if uploaded_file and model:
        if st.button("Process Video"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            
            col_u1, col_u2 = st.columns(2)
            
            with col_u1:
                st.info("Loading frames...")
                frames = load_video_frames_from_file(tfile.name)
                
                with st.spinner("Please wait for a moment. The AI is analyzing your sign..."):
                    # 1. Call the function
                    res = process_prediction_pipeline(frames, model, CLASS_NAMES)
            
                # 2. Extract values safely
                if res:
                    pred = res["pred"]
                    conf = res["conf"]
                    vid_path = res["overlay"]
                    heatmap_path = res["heatmap"]
                    top3 = res["top3"]
                    insights = res["insights"]
                    metrics = res["metrics"]
                    
                    # Store for session state
                    st.session_state.live_last_result = res
                        
                if res and pred:
                    st.success("Analysis Complete!")
                else: 
                    st.error("Processing failed.")

            with col_u2:
                # UPDATED TABS: Renamed Heatmap -> AI Coaching, Added Deep XAI
                u_tabs = st.tabs(["📊 Analysis", "🤖 AI Coaching", "🧠 Deep XAI"])
                
                # --- TAB 1: ANALYSIS ---
                with u_tabs[0]:
                    st.caption("Standard Structure")
                    if vid_path: st.video(vid_path)

                    with st.container(border=True):
                         st.markdown(f"### 🔮 I think you signed: **{pred.replace('_', ' ').title()}**")

                    with st.container(border=True):
                        st.markdown("##### 🎯 AI Confidence Score")
                        if len(top3) >= 1:
                            score = top3[0][1]
                            st.progress(score)
                            if score > 0.9: st.caption("🌟 Excellent Match")
                            elif score > 0.7: st.caption("✅ Good Match")
                            else: st.caption("⚠️ Uncertain / Low Confidence")
                            st.metric("🥇 Top Match", top3[0][0], f"{score:.1%}")

                        if len(top3) >= 2:
                            st.metric("🥈 Alternative", top3[1][0], f"{top3[1][1]:.1%}", delta_color="off")
                        if len(top3) >= 3:
                            st.metric("🥉 Alternative", top3[2][0], f"{top3[2][1]:.1%}", delta_color="off")

                    if dev_mode and metrics:
                        with st.expander("⚡ System Latency (Dev)", expanded=True):
                            k1, k2, k3 = st.columns(3)
                            k1.metric("Preproc", f"{metrics['preprocess']*1000:.0f}ms")
                            k2.metric("Inference", f"{metrics['inference']*1000:.0f}ms")
                            # Handle total time calculation if not in dictionary
                            total_t = metrics.get('total_time', metrics['preprocess'] + metrics['inference'])
                            k3.metric("Total", f"{total_t:.2f}s")

                    st.divider()
                    try:
                        clean_pred = pred.replace("_", " ")
                        trans = GoogleTranslator(source='auto', target=target_lang).translate(clean_pred)
                        st.markdown("#### 🗣️ Translation & Speech")
                        t1, t2 = st.columns(2)
                        with t1: st.info(f"**Original (BM):**\n{clean_pred.title()}")
                        with t2: st.success(f"**Translated ({target_lang}):**\n{trans}")
                        tts = gTTS(trans, lang=target_lang)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                            tts.save(fp.name)
                            st.audio(fp.name, format="audio/mp3")
                    except: pass
                    
                # --- TAB 2: AI COACHING (Formerly Motion Heatmap) ---
                with u_tabs[1]:
                    st.caption("Velocity Trails (Red = Fast)")
                    if heatmap_path: st.video(heatmap_path)
                    
                    if insights:
                        st.divider()
                        st.markdown("##### 🤖 AI Motion Coaching (Beta)")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("Kinetic Energy")
                            st.markdown(f"**{insights.get('energy_text', '-')}**")
                            st.caption(insights.get("energy_tip", ""))
                        with c2:
                            st.caption("Space Usage")
                            st.markdown(f"**{insights.get('size_text', '-')}**")
                            st.caption(insights.get("size_tip", ""))

                # --- TAB 3: DEEP XAI (NEW FEATURE) ---
                with u_tabs[2]: 
                    conf = res["conf"]
                    if conf > 0.4 and "xai_data" in res:
                        xai_data = res["xai_data"]
                        
                        # 1. Compute Saliency
                        saliency = compute_dual_xai(model, xai_data["morph"], xai_data["traj"], 0)
                        
                        # 2. GENERATE NARRATIVE (The "Why")
                        narrative, peak_frame_idx = generate_xai_narrative(saliency, JOINT_NAMES)
                        
                        st.info(f"💡 **AI Insight:** {narrative}")
                        
                        # 3. SHOW THE "DECIDING MOMENT"
                        col_x1, col_x2 = st.columns([1, 1])
                        
                        with col_x1:
                            st.markdown("#### 📸 The Deciding Frame")
                            st.caption("The exact moment the AI made its decision.")
                            peak_img = get_peak_frame_image(xai_data["frames"], peak_frame_idx, saliency, xai_data["skeleton"])
                            if peak_img is not None:
                                st.image(peak_img, channels="BGR", use_container_width=True)
                        
                        with col_x2:
                            st.markdown("#### 🎥 Full Attention Replay")
                            st.caption("Red Glow = High Importance")
                            xai_vid = generate_xai_video(xai_data["frames"], saliency, xai_data["skeleton"])
                            if xai_vid: st.video(xai_vid)

                        # 4. SHOW IMPROVED CHARTS
                        st.divider()
                        st.markdown("#### 📊 Detailed Breakdown")
                        fig = render_xai_charts(saliency, JOINT_NAMES)
                        st.pyplot(fig)
                        
                    else:
                        st.warning("⚠️ Confidence too low for deep analysis (Need > 40%) or XAI data missing.")

# --- TAB 3: ENDLESS PRACTICE MODE ---
with tab_practice:
    st.header("🎮 Endless Practice Mode")
    
    with st.expander("ℹ️ How to Practice", expanded=False):
        st.markdown("""
        1. **Select a Module**: Choose a topic like 'Family' or 'Greetings'.
        2. **See the Word**: The app will give you a word to sign.
        3. **Record & Submit**: Record yourself performing the sign.
        4. **Get Feedback**: See if you got it right and you can move on to the next word!
        """)

    # Category Selector
    # Ensure CATEGORIES exists in your global scope (from previous code)
    if "CATEGORIES" in locals():
        cat_list = list(CATEGORIES.keys())
    else:
        # Fallback if CATEGORIES isn't defined yet
        cat_list = ["General"] 
    
    selected_category = st.selectbox("📚 Select Lesson Module:", cat_list, key="practice_cat_selector")
    
    # Reset if category changes
    if "practice_category" not in st.session_state: st.session_state.practice_category = selected_category
    
    if selected_category != st.session_state.practice_category:
        st.session_state.practice_category = selected_category
        st.session_state.practice_active = False 
        st.rerun()
        
    # Get word pool
    current_word_pool = CATEGORIES[selected_category] if "CATEGORIES" in locals() else ["saya", "anda"]
    
    # --- STATE 1: LOBBY ---
    if not st.session_state.get("practice_active", False):
        st.info(f"Practice **{len(current_word_pool)} words** from the **{selected_category}** module!")
        if st.button("Start Practice"):
            st.session_state.practice_active = True
            st.session_state.practice_streak = 0
            st.session_state.practice_target = random.choice(current_word_pool)
            st.session_state.practice_result_shown = False
            st.session_state.practice_is_recording = False
            st.session_state.practice_last_data = None
            st.rerun()

    # --- STATE 2: ACTIVE GAME ---
    elif st.session_state.practice_active:
        col_header1, col_header2 = st.columns([0.25, 0.75])
        with col_header1:
            st.metric("Streak", f"🔥 {st.session_state.practice_streak}")
        with col_header2:
            st.markdown(f"### Target: :blue-background[{st.session_state.practice_target}]")
            try:
                text_to_speak = st.session_state.practice_target.replace("_", " ")
                tts_target = gTTS(text_to_speak, lang='ms')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp_target:
                    tts_target.save(fp_target.name)
                    st.audio(fp_target.name, format="audio/mp3")
            except: st.caption("Audio unavailable")
        
        col_q1, col_q2 = st.columns([0.6, 0.4])
        
        # --- LEFT COLUMN: CAMERA ---
        with col_q1:
            if not st.session_state.practice_result_shown:
                btn_practice_placeholder = st.empty()
                ctx_q = webrtc_streamer(
                    key="practice-recorder",
                    mode=WebRtcMode.SENDRECV,
                    video_transformer_factory=VideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
                practice_cam_active = ctx_q.state.playing if ctx_q.state else False
                
                with btn_practice_placeholder:
                    btn_q_label = "🔴 Start Recording" if not st.session_state.practice_is_recording else "⏹️ Stop & Submit"
                    if st.button(btn_q_label, key="practice_rec_btn", disabled=not practice_cam_active):
                        if st.session_state.practice_is_recording:
                            st.session_state.practice_is_recording = False
                            st.session_state.practice_waiting = True
                        else:
                            st.session_state.practice_is_recording = True
                            st.session_state.practice_waiting = False
                        st.rerun()
                    if not practice_cam_active: st.caption("⚠️ Please start the camera to practice.")
                
                if ctx_q.video_transformer:
                    ctx_q.video_transformer.recording = st.session_state.practice_is_recording
                    ctx_q.video_transformer.queue = st.session_state.result_queue
            else:
                st.info("Check your results ->")

        # --- RIGHT COLUMN: PROCESSING & RESULTS ---
        with col_q2:
            # 1. PROCESS LOGIC
            if st.session_state.get("practice_waiting", False):
                with st.spinner("Checking your answer..."):
                    try:
                        msg = st.session_state.result_queue.get(timeout=5)
                        if msg['type'] == 'process':
                            q_frames = msg['frames']
                            
                            # [FIX] CALL PIPELINE AND GET DICTIONARY
                            res = process_prediction_pipeline(q_frames, model, CLASS_NAMES)
                            
                            # Logic Check
                            is_correct = (res["pred"] == st.session_state.practice_target)
                            
                            if is_correct: st.session_state.practice_streak += 1
                            else: st.session_state.practice_streak = 0
                            
                            # Add the correctness flag to the result
                            res["correct"] = is_correct
                            
                            # Save EVERYTHING to session state
                            st.session_state.practice_last_data = res
                            
                            st.session_state.practice_waiting = False
                            st.session_state.practice_result_shown = True
                            st.rerun()
                    except queue.Empty:
                        st.error("Timeout. Try again.")
                        st.session_state.practice_waiting = False

            # 2. DISPLAY LOGIC
            if st.session_state.practice_result_shown and st.session_state.practice_last_data:
                data = st.session_state.practice_last_data
                
                # --- FEEDBACK HEADER (Clean & Simple) ---
                if data["correct"]:
                    st.success(f"✅ Correct! I saw **{data['pred']}**")
                    st.balloons()
                else:
                    st.error(f"❌ Incorrect. Target was **{st.session_state.practice_target}**")
                
                # --- VISUAL ANALYSIS TABS ---
                st.markdown("### 🆚 Visual Analysis")
                vc1, vc2, vc3 = st.tabs(["📊 Analysis", "🤖 AI Coaching", "🧠 Deep XAI"])
                
                # === TAB 1: STANDARD ANALYSIS ===
                with vc1:
                    st.caption("Standard Structure")
                    if data.get("overlay"): st.video(data["overlay"])
                    
                    # Card 1: Main Prediction
                    with st.container(border=True):
                         st.markdown(f"### 🔮 I saw: **{data['pred'].replace('_', ' ').title()}**")

                    # Card 2: Confidence Score (UPDATED: Shows Top 3 Here)
                    with st.container(border=True):
                        st.markdown("##### 🎯 AI Confidence Score")
                        top3 = data["top3"]
                        
                        # Loop through Top 3 and display them nicely
                        for i, (name, prob) in enumerate(top3):
                            # Set Label based on rank
                            if i == 0: 
                                label = "🥇 Top Match"
                                color = "green" if data["correct"] else "red" # Red if top match was wrong
                            elif i == 1: 
                                label = "🥈 Alternative"
                                color = "blue"
                            else: 
                                label = "🥉 Alternative"
                                color = "blue"
                            
                            # Display Name and %
                            c_label, c_val = st.columns([0.7, 0.3])
                            with c_label: st.write(f"**{label}:** {name}")
                            with c_val: st.write(f"**{prob:.1%}**")
                            
                            # Display Bar
                            st.progress(prob)
                
                # === TAB 2: AI COACHING ===
                with vc2:
                    st.caption("Motion Analysis")
                    if data.get("heatmap"): st.video(data["heatmap"])
                    
                    if "insights" in data and data["insights"]:
                        ins = data["insights"]
                        st.divider()
                        c1, c2 = st.columns(2)
                        c1.metric("Energy", ins.get('energy_text', '-'))
                        c2.metric("Size", ins.get('size_text', '-'))

                # === TAB 3: DEEP XAI (Now matches Live/Upload features) ===
                with vc3:
                    conf = data["conf"]
                    # Show XAI if we have data
                    if "xai_data" in data and conf > 0.3:
                        xai_data = data["xai_data"]
                        
                        # 1. Compute Saliency
                        saliency = compute_dual_xai(model, xai_data["morph"], xai_data["traj"], 0)
                        
                        # 2. GENERATE NARRATIVE 
                        narrative, peak_frame_idx = generate_xai_narrative(saliency, JOINT_NAMES)
                        
                        st.info(f"💡 **AI Insight:** {narrative}")
                        
                        # 3. SHOW SNAPSHOT & VIDEO
                        col_x1, col_x2 = st.columns([1, 1])
                        with col_x1:
                            st.caption("📸 Deciding Moment")
                            peak_img = get_peak_frame_image(xai_data["frames"], peak_frame_idx, saliency, xai_data["skeleton"])
                            if peak_img is not None:
                                st.image(peak_img, channels="BGR", use_container_width=True)
                        
                        with col_x2:
                            st.caption("🎥 Attention Replay")
                            vid = generate_xai_video(xai_data["frames"], saliency, xai_data["skeleton"])
                            if vid: st.video(vid)
                            
                        # 4. CHARTS
                        st.divider()
                        st.caption("Detailed Finger/Joint Breakdown")
                        fig = render_xai_charts(saliency, JOINT_NAMES)
                        st.pyplot(fig)

                    else:
                        st.warning("Confidence too low (or data missing) to generate Deep XAI.")
                
                st.markdown("---")
                
                # --- NAVIGATION BUTTONS ---
                if not data["correct"]:
                    col_retry, col_next = st.columns(2)
                    with col_retry:
                        if st.button("🔁 Try Again"):
                            st.session_state.practice_result_shown = False
                            st.session_state.practice_last_data = None
                            st.rerun()
                    with col_next:
                        if st.button("➡️ Skip Word"):
                            new_word = random.choice(current_word_pool)
                            while new_word == st.session_state.practice_target and len(current_word_pool) > 1:
                                new_word = random.choice(current_word_pool)
                            st.session_state.practice_target = new_word
                            st.session_state.practice_result_shown = False
                            st.session_state.practice_last_data = None
                            st.rerun()
                else:
                    if st.button("➡️ Next Word"):
                        new_word = random.choice(current_word_pool)
                        while new_word == st.session_state.practice_target and len(current_word_pool) > 1:
                            new_word = random.choice(current_word_pool)
                        st.session_state.practice_target = new_word
                        st.session_state.practice_result_shown = False
                        st.session_state.practice_last_data = None
                        st.rerun()

# ==========================================
# TAB 4: MODEL HEALTH (DYNAMIC FROM .PTH)
# ==========================================
if dev_mode:
    with tabs[3]:
        st.header("🛠️ Model Diagnostics")

        # --- 1. ARCHITECTURE STATS (Calculated Live) ---
        if model:
            total_params = get_model_specs(model)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Parameters", f"{total_params/1e6:.2f}M")
            c2.metric("Input Channels", "3 (x,y,z)")
            c3.metric("Classes", len(CLASS_NAMES))
        
        st.divider()

        # --- 2. FLEXIBLE METRICS DISPLAY (From Metadata) ---
        st.subheader("Checkpoint Metadata")
        
        if model_meta:
            st.caption(f"Extracted from `{os.path.basename(model_path)}`")
            
            # Filter distinct numeric metrics for display
            display_metrics = {k: v for k, v in model_meta.items() if isinstance(v, (int, float, str))}
            
            if display_metrics:
                # Dynamic Grid Layout
                cols = st.columns(4)
                for i, (key, val) in enumerate(display_metrics.items()):
                    # Format float values for better readability
                    display_val = f"{val:.4f}" if isinstance(val, float) else str(val)
                    # Pretty print key names (e.g., 'best_acc' -> 'Best Acc')
                    pretty_key = key.replace('_', ' ').title()
                    cols[i % 4].metric(pretty_key, display_val)
            else:
                st.info("No scalar metrics (accuracy, epoch, etc.) found in this checkpoint.")
                st.json(model_meta) # Dump full meta if complex structure
        else:
            st.warning("No metadata found in this model file. It might be a raw state_dictionary.")

        st.divider()
        
        # The 10 classes with the LOWEST accuracy
        worst_classes = [
            "ada", "sampai", "buang", "baik_2", "sekolah", 
            "baik", "kacau", "baca", "makan", "bas"
        ]
        # Their corresponding scores (0.1 = 10%, 0.9 = 90%)
        worst_scores = [0.00, 0.50, 0.50, 0.38, 0.66, 0.60, 0.80, 0.65, 0.80, 0.80]
        
        # --- VISUALIZATION CODE ---
        st.subheader("⚠️ 10 Worst Classes")
        st.caption("Lower accuracy indicates confusion")
            
        # Create DataFrame for Chart
        df_worst = pd.DataFrame({
           "Class": worst_classes,
            "Accuracy": worst_scores
        }).sort_values("Accuracy", ascending=True) # Sort lowest to highest
            
        # Render Red Bar Chart
        st.bar_chart(df_worst.set_index("Class"), color="#ff4b4b")
            
        # Show Table Data
        st.dataframe(df_worst.style.format({"Accuracy": "{:.1%}"}), use_container_width=True)
    
        st.info("ℹ️ These table data are the static snapshots from the training phase.")