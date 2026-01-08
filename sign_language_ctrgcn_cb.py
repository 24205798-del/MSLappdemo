"""
Skeleton-based Sign Language Recognition with dual-stream CTR-GCN.

This script scans a dataset organized as data_root/class_name/*.mp4,
extracts hand + upper-body skeletons with MediaPipe Holistic, normalizes
to hand-centric (morphology) and body-centric (trajectory) coordinates,
and trains a dual-stream CTR-GCN to classify 90 sign classes.

Dependencies: mediapipe, opencv-python, torch.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

# Reduce noisy TensorFlow/absl logging from MediaPipe before imports (most quiet)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

_CPP_DEVNULL = None
_ORIGINAL_STDERR_FD = None


def redirect_stderr_to_devnull_forever():
    global _CPP_DEVNULL, _ORIGINAL_STDERR_FD
    if _CPP_DEVNULL is None:
        if _ORIGINAL_STDERR_FD is None:
            _ORIGINAL_STDERR_FD = os.dup(2)
        _CPP_DEVNULL = open(os.devnull, "w")
        os.dup2(_CPP_DEVNULL.fileno(), 2)  # stderr only


def restore_stderr_forever():
    global _CPP_DEVNULL, _ORIGINAL_STDERR_FD
    if _ORIGINAL_STDERR_FD is None:
        return
    os.dup2(_ORIGINAL_STDERR_FD, 2)
    os.close(_ORIGINAL_STDERR_FD)
    _ORIGINAL_STDERR_FD = None
    if _CPP_DEVNULL is not None:
        try:
            _CPP_DEVNULL.close()
        except Exception:
            pass
        _CPP_DEVNULL = None


os.environ.setdefault("QUIET_CPP_LOGS", "1")
if os.environ.get("QUIET_CPP_LOGS") == "1":
    redirect_stderr_to_devnull_forever()

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

_GLOBAL_SEED = 0


def seed_everything(seed: int) -> None:
    """Set python/numpy/torch seeds (and cuDNN flags) for reproducibility."""
    global _GLOBAL_SEED
    _GLOBAL_SEED = int(seed)
    random.seed(_GLOBAL_SEED)
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.cuda.manual_seed_all(_GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataloader_worker_init_silence_cpp_logs(worker_id: int):
    """Picklable DataLoader worker_init_fn for spawn-based multiprocessing."""
    if os.environ.get("QUIET_CPP_LOGS", "") == "1":
        redirect_stderr_to_devnull_forever()
    # Reproducibility: make each worker deterministic but different.
    # NOTE: must stay top-level picklable for Windows spawn.
    seed = int(_GLOBAL_SEED) + int(worker_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# -----------------------------
# Quiet logging helper
# -----------------------------


@contextmanager
def suppress_cpp_logs():
    """
    Temporarily redirect stderr to os.devnull to silence C++/glog output
    from MediaPipe/TFLite. Use sparingly around Holistic init and processing.
    """
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

# MediaPipe depends on protobuf's GetPrototype in older releases; protobuf>=5 removed it.
# Add a compatibility shim so newer protobuf still works.
import google.protobuf.message_factory as _message_factory

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
    def _get_prototype(self, descriptor):
        return self.GetMessageClass(descriptor)
    _message_factory.MessageFactory.GetPrototype = _get_prototype

# Silence absl/tensorflow logs
try:
    from absl import logging as absl_logging
    if hasattr(absl_logging, "_warn_preinit_stderr"):
        absl_logging._warn_preinit_stderr = False
    absl_logging.set_verbosity(absl_logging.FATAL)
    absl_logging.set_stderrthreshold("fatal")
except Exception:
    pass

import logging as py_logging
py_logging.getLogger("tensorflow").setLevel(py_logging.CRITICAL)
py_logging.getLogger("mediapipe").setLevel(py_logging.CRITICAL)

# Import mediapipe under suppressed C++ logs (init-time only)
with suppress_cpp_logs():
    import mediapipe as mp


# -----------------------------
# Dataset scanning utilities
# -----------------------------


def scan_dataset(data_root: str) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """
    Walk data_root and collect (video_path, class_name) pairs.
    Returns the list and a mapping class_name -> label_id.
    """
    items: List[Tuple[str, str]] = []
    class_names: List[str] = []
    for entry in sorted(os.listdir(data_root)):
        class_dir = os.path.join(data_root, entry)
        if not os.path.isdir(class_dir):
            continue
        class_names.append(entry)
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(".mp4"):
                items.append((os.path.join(class_dir, fname), entry))
    class_to_id = {c: i for i, c in enumerate(sorted(class_names))}
    return items, class_to_id


# -----------------------------
# Video loading and sampling
# -----------------------------


def load_video_frames(path: str) -> Tuple[List[np.ndarray], float]:
    """
    Read all frames from a video using OpenCV.
    Returns list of RGB frames and the original fps.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    frames: List[np.ndarray] = []
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames, fps


def sample_frames_uniform(frames: Sequence[np.ndarray], target_len: int = 40) -> List[np.ndarray]:
    """
    Uniformly sample exactly `target_len` frames over the full length of the clip.
    Robust for short clips (avoids overly-clumped duplicate indices).
    """
    n = len(frames)
    if target_len <= 0:
        return []
    if n == 0:
        return []
    if n == target_len:
        return list(frames)
    if n == 1:
        return [frames[0] for _ in range(target_len)]

    # For n < target_len, sample in [0, n) with endpoint=False to spread repeats evenly.
    if n < target_len:
        indices = np.linspace(0, n, num=target_len, endpoint=False).astype(np.int64)
        indices = np.clip(indices, 0, n - 1)
    else:
        indices = np.linspace(0, n - 1, num=target_len).astype(np.int64)
        indices[-1] = n - 1
    return [frames[int(i)] for i in indices.tolist()]


# -----------------------------
# MediaPipe Holistic extraction
# -----------------------------

# Pose and hand landmark definitions
POSE_LM = mp.solutions.holistic.PoseLandmark
HAND_LM = mp.solutions.holistic.HandLandmark

# Upper-body pose only (no face landmarks). "torso" is a virtual mid-shoulder node.
POSE_JOINTS: List[Tuple[str, Optional[int]]] = [
    ("torso", None),  # virtual: 0.5*(left_shoulder + right_shoulder)
    ("left_shoulder", POSE_LM.LEFT_SHOULDER),
    ("right_shoulder", POSE_LM.RIGHT_SHOULDER),
    ("left_elbow", POSE_LM.LEFT_ELBOW),
    ("right_elbow", POSE_LM.RIGHT_ELBOW),
    ("left_wrist", POSE_LM.LEFT_WRIST),
    ("right_wrist", POSE_LM.RIGHT_WRIST),
]

HAND_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

# Use hand-centric names without left_/right_ prefixes; the stream prefix (left_hand/right_hand) distinguishes them.
LEFT_HAND_JOINTS: List[Tuple[str, int]] = [(n, lm) for n, lm in zip(HAND_NAMES, HAND_LM)]
RIGHT_HAND_JOINTS: List[Tuple[str, int]] = [(n, lm) for n, lm in zip(HAND_NAMES, HAND_LM)]

JOINT_LIST: List[Tuple[str, str, Optional[int]]] = (
    [("pose", name, idx) for name, idx in POSE_JOINTS]
    + [("left_hand", name, idx) for name, idx in LEFT_HAND_JOINTS]
    + [("right_hand", name, idx) for name, idx in RIGHT_HAND_JOINTS]
)

JOINT_NAMES: List[str] = [f"{src}:{name}" for src, name, _ in JOINT_LIST]
JOINT_INDEX: Dict[str, int] = {name: i for i, name in enumerate(JOINT_NAMES)}


def _interpolate_missing(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs in (T, J, C) by linear interpolation across time."""
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


def _smooth_temporal(arr: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple moving average smoothing over time."""
    if window <= 1:
        return arr
    pad = window // 2
    padded = np.pad(arr, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.zeros_like(arr)
    for jj in range(arr.shape[1]):
        for cc in range(arr.shape[2]):
            smoothed[:, jj, cc] = np.convolve(padded[:, jj, cc], kernel, mode="valid")
    return smoothed


class HolisticExtractor:
    """Wrapper to run MediaPipe Holistic once and reuse for all frames."""

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        debug: bool = False,
    ):
        self.debug = bool(debug)
        if self.debug:
            self.holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=int(model_complexity),
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            with suppress_cpp_logs():
                self.holistic = mp.solutions.holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=int(model_complexity),
                    enable_segmentation=False,
                    refine_face_landmarks=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )

    def _process_frame(self, frame: np.ndarray, frame_index: Optional[int] = None):
        try:
            if self.debug:
                return self.holistic.process(frame)
            with suppress_cpp_logs():
                return self.holistic.process(frame)
        except Exception as e:
            if self.debug:
                idx = frame_index if frame_index is not None else "unknown"
                print(f"[ERROR] MediaPipe exception at frame {idx}: {repr(e)}", file=sys.stderr)
                raise
            return None

    def __call__(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        t_len = len(frames)
        j_len = len(JOINT_LIST)
        skeleton = np.full((t_len, j_len, 3), np.nan, dtype=np.float32)
        torso_idx = JOINT_INDEX.get("pose:torso")
        l_shoulder_idx = JOINT_INDEX.get("pose:left_shoulder")
        r_shoulder_idx = JOINT_INDEX.get("pose:right_shoulder")
        for t, frame in enumerate(frames):
            results = self._process_frame(frame, frame_index=t)
            if results is None:
                continue
            try:
                # Pose upper-body joints
                if results.pose_landmarks:
                    for local_idx, (_, idx_enum) in enumerate(POSE_JOINTS):
                        if idx_enum is None:
                            continue
                        lm = results.pose_landmarks.landmark[idx_enum]
                        skeleton[t, local_idx] = [lm.x, lm.y, lm.z]
                    # Virtual torso: mid-shoulder (no face landmarks involved)
                    if torso_idx is not None and l_shoulder_idx is not None and r_shoulder_idx is not None:
                        l_sh = skeleton[t, l_shoulder_idx]
                        r_sh = skeleton[t, r_shoulder_idx]
                        if not (np.isnan(l_sh).any() or np.isnan(r_sh).any()):
                            skeleton[t, torso_idx] = 0.5 * (l_sh + r_sh)
                        elif not np.isnan(l_sh).any():
                            skeleton[t, torso_idx] = l_sh
                        elif not np.isnan(r_sh).any():
                            skeleton[t, torso_idx] = r_sh
                # Left hand
                if results.left_hand_landmarks:
                    for (name, idx_enum), joint_idx in zip(
                        LEFT_HAND_JOINTS,
                        range(len(POSE_JOINTS), len(POSE_JOINTS) + len(LEFT_HAND_JOINTS)),
                    ):
                        lm = results.left_hand_landmarks.landmark[idx_enum]
                        skeleton[t, joint_idx] = [lm.x, lm.y, lm.z]
                # Right hand
                if results.right_hand_landmarks:
                    offset = len(POSE_JOINTS) + len(LEFT_HAND_JOINTS)
                    for (name, idx_enum), joint_idx in zip(
                        RIGHT_HAND_JOINTS,
                        range(offset, offset + len(RIGHT_HAND_JOINTS)),
                    ):
                        lm = results.right_hand_landmarks.landmark[idx_enum]
                        skeleton[t, joint_idx] = [lm.x, lm.y, lm.z]
            except Exception as e:
                if self.debug:
                    print(f"[ERROR] MediaPipe exception at frame {t}: {repr(e)}", file=sys.stderr)
                    raise
                continue
        skeleton = _interpolate_missing(skeleton)
        skeleton = _smooth_temporal(skeleton, window=3)
        return skeleton


# -----------------------------
# Normalization
# -----------------------------


def normalize_dual_stream(skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build morphology (hand-centric) and trajectory (body-centric) versions.
    skeleton: (T, J, 3)
    """
    t, j, c = skeleton.shape
    morph = skeleton.copy()
    traj = skeleton.copy()

    l_shoulder = JOINT_INDEX.get("pose:left_shoulder")
    r_shoulder = JOINT_INDEX.get("pose:right_shoulder")
    torso = JOINT_INDEX.get("pose:torso")
    l_wrist_pose = JOINT_INDEX.get("pose:left_wrist")
    r_wrist_pose = JOINT_INDEX.get("pose:right_wrist")
    l_hand_wrist = JOINT_INDEX.get("left_hand:wrist")
    r_hand_wrist = JOINT_INDEX.get("right_hand:wrist")
    l_middle_tip = JOINT_INDEX.get("left_hand:middle_tip")
    r_middle_tip = JOINT_INDEX.get("right_hand:middle_tip")

    morph_out = np.zeros_like(morph)
    traj_out = np.zeros_like(traj)

    for idx in range(t):
        frame = skeleton[idx]

        # Torso origin: prefer virtual torso (mid-shoulder), fallback to shoulders.
        torso_node = frame[torso] if torso is not None else np.array([np.nan, np.nan, np.nan])
        l_sh = frame[l_shoulder] if l_shoulder is not None else np.array([np.nan, np.nan, np.nan])
        r_sh = frame[r_shoulder] if r_shoulder is not None else np.array([np.nan, np.nan, np.nan])
        if not (np.isnan(l_sh).any() or np.isnan(r_sh).any()):
            shoulder_width = np.linalg.norm(l_sh - r_sh) + 1e-6
        else:
            shoulder_width = 1.0

        if not np.isnan(torso_node).any():
            torso_origin = torso_node
        elif not (np.isnan(l_sh).any() or np.isnan(r_sh).any()):
            torso_origin = 0.5 * (l_sh + r_sh)
        elif not np.isnan(l_sh).any():
            torso_origin = l_sh
        elif not np.isnan(r_sh).any():
            torso_origin = r_sh
        else:
            torso_origin = np.zeros(3, dtype=np.float32)

        # Trajectory: center on torso and scale by shoulder width
        traj_centered = frame - torso_origin
        traj_out[idx] = traj_centered / shoulder_width

        # Hand-centric: prefer right hand if valid, else left, else torso
        r_wrist = frame[r_hand_wrist] if r_hand_wrist is not None else np.array([np.nan, np.nan, np.nan])
        l_wrist = frame[l_hand_wrist] if l_hand_wrist is not None else np.array([np.nan, np.nan, np.nan])
        origin = r_wrist
        if np.isnan(origin).any():
            origin = l_wrist
        if np.isnan(origin).any():
            origin = frame[r_wrist_pose] if r_wrist_pose is not None else origin
        if np.isnan(origin).any():
            origin = frame[l_wrist_pose] if l_wrist_pose is not None else origin
        if np.isnan(origin).any():
            origin = torso_origin

        # Hand size for scaling (wrist to middle fingertip)
        hand_scale = 1.0
        if not np.isnan(r_wrist).any() and r_middle_tip is not None and not np.isnan(frame[r_middle_tip]).any():
            hand_scale = np.linalg.norm(frame[r_middle_tip] - r_wrist) + 1e-6
        elif not np.isnan(l_wrist).any() and l_middle_tip is not None and not np.isnan(frame[l_middle_tip]).any():
            hand_scale = np.linalg.norm(frame[l_middle_tip] - l_wrist) + 1e-6
        morph_centered = frame - origin
        morph_out[idx] = morph_centered / hand_scale

    return morph_out, traj_out


# -----------------------------
# PyTorch Dataset
# -----------------------------

_CACHE_VERSION = "stgcn_cache_v1"


def _make_cache_path(
    cache_dir: str,
    video_path: str,
    target_len: int,
    extractor_cfg: Dict[str, object],
) -> str:
    """
    Cache key: hash(video_path + mtime/size + target_len + extractor settings + joint list + version).
    """
    abs_path = os.path.normcase(os.path.abspath(video_path))
    try:
        st = os.stat(video_path)
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        size = int(st.st_size)
    except OSError:
        mtime_ns = 0
        size = 0
    payload = {
        "v": _CACHE_VERSION,
        "path": abs_path,
        "mtime_ns": mtime_ns,
        "size": size,
        "target_len": int(target_len),
        "extractor": extractor_cfg,
        "joints": JOINT_NAMES,
    }
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.npz")


def _atomic_save_npz(path: str, **arrays: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Ensure the temporary file ends with .npz; otherwise NumPy will append it implicitly.
    tmp_path = f"{path}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def _atomic_torch_save(path: str, obj: object) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.pth"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


class SignDataset(Dataset):
    """
    Dataset that loads videos, extracts skeletons, and returns dual-stream tensors.
    Output tensors: morphology (C, T, J), trajectory (C, T, J), label (int).
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        target_len: int = 40,
        holistic_extractor: HolisticExtractor = None,
        holistic_cfg: Optional[Dict[str, object]] = None,
        cache_dir: str = "",
    ):
        self.samples = samples
        self.target_len = target_len
        self._holistic = holistic_extractor  # created lazily per worker if None
        self.holistic_cfg = holistic_cfg or {}
        self.cache_dir = cache_dir.strip()
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        cache_path = ""
        if self.cache_dir:
            cache_path = _make_cache_path(self.cache_dir, video_path, self.target_len, self.holistic_cfg)
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path)
                    morph = data["morph"].astype(np.float32, copy=False)
                    traj = data["traj"].astype(np.float32, copy=False)
                    data.close()
                    morph_t = torch.from_numpy(morph).permute(2, 0, 1).float()  # (C, T, J)
                    traj_t = torch.from_numpy(traj).permute(2, 0, 1).float()
                    return morph_t, traj_t, torch.tensor(label, dtype=torch.long)
                except Exception:
                    # Corrupted cache; fall back to recompute and overwrite.
                    pass

        if self._holistic is None:
            self._holistic = HolisticExtractor(**self.holistic_cfg)

        frames, _ = load_video_frames(video_path)
        if len(frames) == 0:
            # Extremely defensive: keep shapes consistent even if video can't be read.
            morph = np.zeros((self.target_len, len(JOINT_NAMES), 3), dtype=np.float32)
            traj = np.zeros_like(morph)
        else:
            frames = sample_frames_uniform(frames, target_len=self.target_len)
            try:
                skeleton = self._holistic(frames)
                morph, traj = normalize_dual_stream(skeleton)
            except Exception:
                print(f"[ERROR] MediaPipe failed on video: {video_path}", file=sys.stderr)
                raise

        if cache_path:
            try:
                _atomic_save_npz(cache_path, morph=morph.astype(np.float32), traj=traj.astype(np.float32))
            except Exception:
                pass
        morph_t = torch.from_numpy(morph).permute(2, 0, 1).float()  # (C, T, J)
        traj_t = torch.from_numpy(traj).permute(2, 0, 1).float()
        return morph_t, traj_t, torch.tensor(label, dtype=torch.long)


# -----------------------------
# Data splitting & imbalance
# -----------------------------


def stratified_split(
    items: List[Tuple[str, str]],
    class_to_id: Dict[str, int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Stratified split (train/val/test) preserving class distribution.
    """
    per_class: Dict[str, List[str]] = defaultdict(list)
    for path, cname in items:
        per_class[cname].append(path)

    train, val, test = [], [], []
    for cname, paths in per_class.items():
        random.shuffle(paths)
        n = len(paths)
        if n >= 3:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            if n_train + n_val >= n:
                n_train = max(1, n_train - 1)
            n_test = max(1, n - n_train - n_val)
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train, n_val, n_test = 1, 0, 0
        splits = {
            "train": paths[:n_train],
            "val": paths[n_train : n_train + n_val],
            "test": paths[n_train + n_val : n_train + n_val + n_test],
        }
        for p in splits["train"]:
            train.append((p, class_to_id[cname]))
        for p in splits["val"]:
            val.append((p, class_to_id[cname]))
        for p in splits["test"]:
            test.append((p, class_to_id[cname]))
    return train, val, test


def compute_class_weights(samples: List[Tuple[str, int]], num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    """
    counts = torch.zeros(num_classes, dtype=torch.float)
    for _, label in samples:
        counts[label] += 1
    weights = 1.0 / torch.sqrt(torch.clamp(counts, min=1.0))
    weights = weights / weights.sum() * num_classes
    return weights


def make_weighted_sampler(
    samples: List[Tuple[str, int]],
    num_classes: int,
    generator: Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler for class-imbalanced training.
    Weights are inverse-frequency per class (replacement sampling).
    """
    counts = torch.zeros(num_classes, dtype=torch.float)
    labels = torch.tensor([label for _, label in samples], dtype=torch.long)
    for label in labels.tolist():
        counts[label] += 1.0
    class_w = 1.0 / counts.clamp(min=1.0)
    sample_w = class_w[labels].double()
    try:
        return WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True, generator=generator)
    except TypeError:
        return WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)


# -----------------------------
# Graph construction
# -----------------------------


def build_adjacency() -> np.ndarray:
    """
    Build adjacency matrix for pose + both hands.
    """
    j = len(JOINT_NAMES)
    A = np.zeros((j, j), dtype=np.float32)

    def connect(a: str, b: str):
        ia, ib = JOINT_INDEX[a], JOINT_INDEX[b]
        A[ia, ib] = 1.0
        A[ib, ia] = 1.0

    # Pose chain
    connect("pose:torso", "pose:left_shoulder")
    connect("pose:torso", "pose:right_shoulder")
    connect("pose:left_shoulder", "pose:right_shoulder")
    connect("pose:left_shoulder", "pose:left_elbow")
    connect("pose:left_elbow", "pose:left_wrist")
    connect("pose:right_shoulder", "pose:right_elbow")
    connect("pose:right_elbow", "pose:right_wrist")

    # Link pose wrists to hand wrists
    connect("pose:left_wrist", "left_hand:wrist")
    connect("pose:right_wrist", "right_hand:wrist")

    # Hand connectivity (MediaPipe hand bones)
    hand_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]
    # Left hand
    offset_l = len(POSE_JOINTS)
    for a, b in hand_edges:
        connect(JOINT_NAMES[offset_l + a], JOINT_NAMES[offset_l + b])
    # Right hand
    offset_r = len(POSE_JOINTS) + len(LEFT_HAND_JOINTS)
    for a, b in hand_edges:
        connect(JOINT_NAMES[offset_r + a], JOINT_NAMES[offset_r + b])

    # Normalize adjacency (symmetric)
    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-6)))
    A_norm = deg_inv_sqrt @ A @ deg_inv_sqrt
    return A_norm.astype(np.float32)


# -----------------------------
# CTR-GCN blocks and model
# -----------------------------


class CTRGC(nn.Module):
    """
    Channel-wise Topology Refinement Graph Convolution (CTR-GC).
    Uses a fixed adjacency prior with a learnable shared refinement plus
    channel-wise dynamic correlations.
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
        # x: (N, C, T, V)
        n, _, t, v = x.shape
        g = self.conv_g(x)
        theta = self.conv_theta(x).mean(dim=2)  # (N, C_int, V)
        phi = self.conv_phi(x).mean(dim=2)  # (N, C_int, V)
        rel = torch.tanh(theta.unsqueeze(-1) - phi.unsqueeze(-2))  # (N, C_int, V, V)
        rel = self.conv_rel(rel)  # (N, C_out, V, V)
        A = self.A.to(rel.dtype).to(rel.device)
        A = A.view(1, 1, v, v) + self.PA.view(1, 1, v, v)
        A_dyn = A + self.alpha.view(1, 1, 1, 1) * rel
        out = torch.einsum("nctv,ncuv->nctu", g, A_dyn)
        return out


class CTRGCNBlock(nn.Module):
    """CTR-GC + temporal conv + residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        kernel_size_t: int = 9,
        dropout: float = 0.5,
    ):
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
        # x: (N, C, T, V)
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
    """Stacked CTR-GCN blocks for a single stream."""

    def __init__(self, A: np.ndarray, hidden_dim: int = 96, num_blocks: int = 4, dropout: float = 0.5):
        super().__init__()
        A_tensor = torch.tensor(A, dtype=torch.float32)
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
    """
    Two CTR-GCN branches:
      - morphology stream: hand-centric coordinates
      - trajectory stream: torso-centric coordinates
    Outputs concatenated features for classification.
    """

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
        # Inputs: (N, C, T, V)
        m = self.morph_stream(morph)
        t = self.traj_stream(traj)
        m = self.morph_drop(m)
        t = self.traj_drop(t)
        m_feat = m.mean(dim=[2, 3])
        t_feat = t.mean(dim=[2, 3])
        feat = torch.cat([m_feat, t_feat], dim=1)
        logits = self.classifier(feat)
        return logits


# -----------------------------
# Training utilities
# -----------------------------


def is_main_process() -> bool:
    """
    Best-effort check for "main" process (rank 0) when launched under DDP/Slurm.
    """
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank()) == 0
    except Exception:
        pass
    for env_key in ("RANK", "SLURM_PROCID"):
        if env_key in os.environ:
            try:
                return int(os.environ[env_key]) == 0
            except ValueError:
                return True
    return True


def _sanitize_tb_tag(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned if cleaned else "unknown"


def confusion_matrix_from_preds(labels: torch.Tensor, preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute confusion matrix with rows=true labels, cols=pred labels.
    """
    labels = labels.to(torch.int64).view(-1)
    preds = preds.to(torch.int64).view(-1)
    k = labels * int(num_classes) + preds
    cm = torch.bincount(k, minlength=int(num_classes) * int(num_classes))
    return cm.reshape(int(num_classes), int(num_classes)).to(torch.long)


def classification_metrics_from_cm(cm: torch.Tensor, eps: float = 1e-12) -> Dict[str, object]:
    """
    Return epoch-level metrics from a confusion matrix (rows=true, cols=pred).
    Metrics: top1_acc, macro_f1, weighted_f1, balanced_acc, per-class precision/recall/f1/support.
    """
    cm_f = cm.to(torch.float32)
    tp = torch.diag(cm_f)
    support = cm_f.sum(dim=1)  # true counts
    pred_support = cm_f.sum(dim=0)  # predicted counts

    precision = tp / pred_support.clamp(min=1.0)
    recall = tp / support.clamp(min=1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=eps)

    total = cm_f.sum().clamp(min=1.0)
    top1_acc = float(tp.sum().div(total).item())

    macro_f1 = float(f1.mean().item())
    balanced_acc = float(recall.mean().item())

    weighted_f1 = float((f1 * support).sum().div(support.sum().clamp(min=1.0)).item())

    return {
        "top1_acc": top1_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_acc": balanced_acc,
        "per_class_precision": precision.detach().cpu(),
        "per_class_recall": recall.detach().cpu(),
        "per_class_f1": f1.detach().cpu(),
        "support": support.detach().cpu().to(torch.long),
    }


def worst_k_by_recall(per_class_recall: torch.Tensor, support: torch.Tensor, k: int) -> List[int]:
    """
    Return class ids with lowest recall among classes with support>0.
    """
    k = int(k)
    if k <= 0:
        return []
    support = support.view(-1)
    per_class_recall = per_class_recall.view(-1)
    valid = support > 0
    if not bool(valid.any()):
        return []
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    recalls = per_class_recall[valid_idx]
    order = torch.argsort(recalls)  # ascending
    worst = valid_idx[order[: min(k, int(order.numel()))]]
    return worst.tolist()


def format_worst_k_table(
    class_ids: Sequence[int],
    class_names: Sequence[str],
    support: torch.Tensor,
    recall: torch.Tensor,
    precision: torch.Tensor,
    f1: torch.Tensor,
) -> str:
    lines = [
        "|rank|class_id|class_name|support|recall|precision|f1|",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for rank, cid in enumerate(class_ids, start=1):
        cname = class_names[cid] if cid < len(class_names) else str(cid)
        sup = int(support[cid].item())
        r = float(recall[cid].item())
        p = float(precision[cid].item())
        f = float(f1[cid].item())
        lines.append(f"|{rank}|{cid}|{cname}|{sup}|{r:.3f}|{p:.3f}|{f:.3f}|")
    return "\n".join(lines)


def plot_confusion_matrix(
    cm: torch.Tensor,
    class_names: Sequence[str],
    normalize: bool = True,
    max_classes: int = 90,
    title: str = "",
):
    """
    Matplotlib confusion matrix helper. Returns a Figure or None if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    cm_np = cm.detach().cpu().numpy()
    n = int(cm_np.shape[0])
    if n > int(max_classes):
        cm_np = cm_np[: int(max_classes), : int(max_classes)]
        n = int(max_classes)
        class_names = list(class_names)[: int(max_classes)]

    if normalize:
        denom = cm_np.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1
        cm_disp = cm_np.astype(np.float32) / denom.astype(np.float32)
        vmin, vmax = 0.0, 1.0
    else:
        cm_disp = cm_np
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(8.5, 8.0), dpi=120)
    im = ax.imshow(cm_disp, interpolation="nearest", cmap="Blues", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)

    if n <= 30:
        tick = np.arange(n)
        ax.set_xticks(tick)
        ax.set_yticks(tick)
        ax.set_xticklabels(list(class_names)[:n], rotation=90, fontsize=6)
        ax.set_yticklabels(list(class_names)[:n], fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig


def estimate_cache_hit_rate(
    samples: Sequence[Tuple[str, int]],
    cache_dir: str,
    target_len: int,
    extractor_cfg: Dict[str, object],
) -> float:
    if not cache_dir:
        return 0.0
    hits = 0
    total = int(len(samples))
    for video_path, _ in samples:
        cache_path = _make_cache_path(cache_dir, video_path, target_len, extractor_cfg)
        if os.path.exists(cache_path):
            hits += 1
    return float(hits / total) if total > 0 else 0.0


def batch_quality_metrics(
    morph: torch.Tensor,
    traj: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Lightweight data/extraction quality diagnostics on (B,C,T,J) tensors.
    Returns scalar tensors on the same device (convert to Python floats only when needed).
    """

    def near_zero_pct(x: torch.Tensor) -> torch.Tensor:
        return (x.abs() < eps).to(torch.float32).mean().mul(100.0)

    def motion_energy(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # (C, T, J)
            x = x.unsqueeze(0)
        if x.dim() != 4 or int(x.size(2)) < 2:
            return x.new_zeros(())
        diff = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_flat = diff.permute(0, 2, 1, 3).contiguous().flatten(start_dim=2)
        l2 = torch.norm(diff_flat, p=2, dim=2)
        return l2.mean()

    with torch.no_grad():
        nan_count = torch.isnan(morph).sum() + torch.isnan(traj).sum()
        return {
            "morph_near_zero_pct": near_zero_pct(morph),
            "traj_near_zero_pct": near_zero_pct(traj),
            "morph_motion_energy": motion_energy(morph),
            "traj_motion_energy": motion_energy(traj),
            "nan_count": nan_count.to(torch.float32),
        }


def global_grad_norm_l2(parameters) -> float:
    total_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2)
        total_sq += float(param_norm.item() ** 2)
    return math.sqrt(total_sq)


def log_epoch_metrics_to_tensorboard(
    writer: SummaryWriter,
    split: str,
    metrics: Dict[str, object],
    epoch: int,
    class_names: Sequence[str],
    worst_k: int,
) -> List[int]:
    """
    Log epoch-level metrics for a split and (optionally) worst_k classes by recall.
    Returns the selected worst_k class ids (may be empty).
    """
    writer.add_scalar(f"{split}/loss", float(metrics.get("loss", 0.0)), epoch)
    writer.add_scalar(f"{split}/top1_acc", float(metrics["top1_acc"]), epoch)
    writer.add_scalar(f"{split}/macro_f1", float(metrics["macro_f1"]), epoch)
    writer.add_scalar(f"{split}/weighted_f1", float(metrics["weighted_f1"]), epoch)
    writer.add_scalar(f"{split}/balanced_acc", float(metrics["balanced_acc"]), epoch)

    worst_ids: List[int] = []
    worst_k = int(worst_k)
    if worst_k > 0:
        precision = metrics["per_class_precision"]
        recall = metrics["per_class_recall"]
        f1 = metrics["per_class_f1"]
        support = metrics["support"]
        worst_ids = worst_k_by_recall(recall, support, worst_k)
        writer.add_text(
            f"{split}/worst_k_by_recall",
            format_worst_k_table(worst_ids, class_names, support, recall, precision, f1),
            epoch,
        )
        for cid in worst_ids:
            cname = class_names[cid] if cid < len(class_names) else str(cid)
            tag = f"{int(cid):03d}_{_sanitize_tb_tag(cname)}"
            writer.add_scalar(f"{split}/worst_recall/{tag}", float(recall[cid].item()), epoch)
            writer.add_scalar(f"{split}/worst_precision/{tag}", float(precision[cid].item()), epoch)
            writer.add_scalar(f"{split}/worst_support/{tag}", float(support[cid].item()), epoch)
            writer.add_scalar(f"{split}/worst_f1/{tag}", float(f1[cid].item()), epoch)
    return worst_ids


class WeightedLabelSmoothingCrossEntropy(nn.Module):
    """
    Fallback for older PyTorch without nn.CrossEntropyLoss(label_smoothing=...).
    Supports optional class weights for the hard-label term.
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        nll = -log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        if self.weight is not None:
            nll = nll * self.weight[target]
        if self.label_smoothing <= 0.0:
            return nll.mean()
        if self.weight is None:
            smooth = -log_probs.mean(dim=1)
        else:
            w = self.weight / self.weight.sum().clamp(min=1e-12)
            smooth = -(log_probs * w.unsqueeze(0)).sum(dim=1)
        loss = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth
        return loss.mean()


def compute_class_counts(samples: List[Tuple[str, int]], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, label in samples:
        counts[int(label)] += 1
    return counts


def compute_cb_class_weights(counts: torch.Tensor, beta: float, weight_norm: str = "mean") -> torch.Tensor:
    counts_f = counts.to(dtype=torch.float32)
    beta = float(beta)
    weights = torch.zeros_like(counts_f)
    nonzero = counts_f > 0
    if nonzero.any():
        weights[nonzero] = (1.0 - beta) / (1.0 - torch.pow(beta, counts_f[nonzero]))
    if weight_norm == "sum":
        denom = weights.sum().clamp(min=1e-12)
        weights = weights * (float(len(weights)) / denom)
    else:
        denom = weights.mean().clamp(min=1e-12)
        weights = weights / denom
    return weights


def make_cb_criterion(
    weight: Optional[torch.Tensor],
    label_smoothing: float,
) -> nn.Module:
    """Create class-balanced CrossEntropy loss with optional label smoothing."""
    label_smoothing = float(label_smoothing)
    try:
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    except TypeError:
        if label_smoothing > 0.0:
            return WeightedLabelSmoothingCrossEntropy(weight=weight, label_smoothing=label_smoothing)
        return nn.CrossEntropyLoss(weight=weight)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    num_classes: int,
    epoch: int,
    args,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
):
    model.train()
    total_loss = 0.0
    num_samples = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    quality_sum = {
        "morph_near_zero_pct": torch.zeros((), device=device, dtype=torch.float32),
        "traj_near_zero_pct": torch.zeros((), device=device, dtype=torch.float32),
        "morph_motion_energy": torch.zeros((), device=device, dtype=torch.float32),
        "traj_motion_energy": torch.zeros((), device=device, dtype=torch.float32),
        "nan_count": torch.zeros((), device=device, dtype=torch.float32),
    }
    num_batches = 0
    step_time_sum = 0.0
    data_time_sum = 0.0
    epoch_start = time.perf_counter()
    end = epoch_start

    log_every = int(getattr(args, "tb_log_every_steps", 0))

    for morph, traj, labels in tqdm(loader, desc="Train", leave=False, file=sys.stdout):
        data_time = time.perf_counter() - end
        iter_start = time.perf_counter()

        morph = morph.to(device, non_blocking=True)
        traj = traj.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(morph, traj)
        loss = criterion(logits, labels)
        loss.backward()

        should_log = writer is not None and log_every > 0 and (global_step % log_every == 0)
        grad_norm = None
        confidence_mean = None
        if should_log:
            grad_norm = global_grad_norm_l2(model.parameters())
            with torch.no_grad():
                confidence_mean = float(logits.softmax(dim=1).max(dim=1).values.mean().item())

        optimizer.step()

        bs = int(labels.size(0))
        total_loss += float(loss.item()) * bs
        num_samples += bs

        preds = logits.argmax(dim=1).detach().cpu()
        labels_cpu = labels.detach().cpu()
        cm += confusion_matrix_from_preds(labels_cpu, preds, num_classes)

        q = batch_quality_metrics(morph, traj, eps=1e-6)
        for k in quality_sum:
            quality_sum[k] += q[k].to(quality_sum[k].dtype)

        step_time = float(time.perf_counter() - iter_start)

        if should_log:
            lr = float(optimizer.param_groups[0]["lr"])
            writer.add_scalar("train/loss_step", float(loss.item()), global_step)
            writer.add_scalar("train/lr", lr, global_step)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
            writer.add_scalar("train/step_time_sec", float(step_time), global_step)
            writer.add_scalar("train/data_time_sec", float(data_time), global_step)
            if device.type == "cuda" and torch.cuda.is_available():
                writer.add_scalar("gpu/mem_allocated_mb", float(torch.cuda.memory_allocated() / (1024**2)), global_step)
                writer.add_scalar("gpu/mem_reserved_mb", float(torch.cuda.memory_reserved() / (1024**2)), global_step)
            if confidence_mean is not None:
                writer.add_scalar("train/confidence_mean", float(confidence_mean), global_step)
            writer.add_scalar("data/morph_near_zero_pct", float(q["morph_near_zero_pct"].item()), global_step)
            writer.add_scalar("data/traj_near_zero_pct", float(q["traj_near_zero_pct"].item()), global_step)
            writer.add_scalar("data/morph_motion_energy", float(q["morph_motion_energy"].item()), global_step)
            writer.add_scalar("data/traj_motion_energy", float(q["traj_motion_energy"].item()), global_step)
            writer.add_scalar("data/nan_count", float(q["nan_count"].item()), global_step)

        end = time.perf_counter()

        num_batches += 1
        step_time_sum += float(step_time)
        data_time_sum += float(data_time)

        global_step += 1

    avg_loss = total_loss / max(1, num_samples)
    epoch_time = float(time.perf_counter() - epoch_start)
    quality_avg = {k: float((quality_sum[k] / max(1, num_batches)).item()) for k in quality_sum}
    timing = {
        "epoch_time_sec": epoch_time,
        "avg_step_time_sec": float(step_time_sum / max(1, num_batches)),
        "avg_data_time_sec": float(data_time_sum / max(1, num_batches)),
        "samples_per_sec": float(num_samples / max(1e-12, epoch_time)),
        "num_samples": int(num_samples),
        "num_batches": int(num_batches),
    }
    return avg_loss, cm, quality_avg, timing, global_step


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
    stage: str = "Eval",
    return_outputs: bool = False,
    return_probs: bool = False,
):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    preds_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    probs_all: List[torch.Tensor] = []
    for morph, traj, labels in tqdm(loader, desc=stage, leave=False, file=sys.stdout):
        morph = morph.to(device, non_blocking=True)
        traj = traj.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(morph, traj)
        loss = criterion(logits, labels)
        bs = int(labels.size(0))
        total_loss += float(loss.item()) * bs
        num_samples += bs

        preds = logits.argmax(dim=1).detach().cpu()
        labels_cpu = labels.detach().cpu()
        cm += confusion_matrix_from_preds(labels_cpu, preds, num_classes)

        if return_outputs:
            preds_all.append(preds)
            labels_all.append(labels_cpu)
            if return_probs:
                probs_all.append(logits.softmax(dim=1).detach().cpu())

    avg_loss = total_loss / max(1, num_samples)
    metrics = classification_metrics_from_cm(cm)
    metrics["loss"] = float(avg_loss)
    metrics["cm"] = cm

    outputs = None
    if return_outputs:
        outputs = {
            "preds": torch.cat(preds_all, dim=0) if preds_all else torch.empty((0,), dtype=torch.long),
            "labels": torch.cat(labels_all, dim=0) if labels_all else torch.empty((0,), dtype=torch.long),
        }
        if return_probs:
            outputs["probs"] = torch.cat(probs_all, dim=0) if probs_all else torch.empty((0, num_classes))
    return metrics, outputs


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser(description="Dual-stream CTR-GCN for skeleton-based sign language recognition.")
    parser.add_argument("--data_root", type=str, default="BIM Dataset V3", help="Path to dataset root (class_name/*.mp4).")
    # Updated defaults for 90-class isolated sign recognition (no compute limit).
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "cosine", "step"],
        default="cosine",
        help='LR scheduler: "cosine" (default), "step", or "none".',
    )
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--target_len", type=int, default=80, help="Number of frames uniformly sampled per clip.")
    parser.add_argument("--hidden_dim", type=int, default=96, help="Base hidden dim for CTR-GCN.")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of CTR-GCN blocks per stream.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout used in blocks/streams/classifier.")
    parser.add_argument(
        "--imbalance_strategy",
        type=str,
        choices=["loss_weight", "sampler", "both"],
        default="both",
        help="Imbalance handling for training: class-weighted loss, weighted sampler, or both.",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="CrossEntropy label_smoothing.")
    parser.add_argument("--cb_beta", type=float, default=0.9999, help="CB loss beta for effective number of samples.")
    parser.add_argument(
        "--cb_weight_norm",
        type=str,
        choices=["sum", "mean"],
        default="mean",
        help="Normalization for CB loss class weights.",
    )
    parser.add_argument(
        "--disable_sampler",
        action="store_true",
        help="Disable WeightedRandomSampler even if imbalance_strategy enables it.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=20,
        help="Stop if val macro-F1 doesn't improve for this many epochs (0 disables).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Optional cache directory for extracted skeletons (empty disables).",
    )
    # MediaPipe Holistic quality knobs (hands+pose only; no face landmarks used).
    parser.add_argument("--model_complexity", type=int, default=2, help="MediaPipe Holistic model_complexity (0/1/2).")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument(
        "--debug_mediapipe",
        action="store_true",
        help="If set, do not suppress MediaPipe exceptions; crash with stack trace.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--log_dir", type=str, default="runs/ctrgcn_cb", help="TensorBoard log directory (empty to disable).")
    parser.add_argument("--tb_flush_secs", type=int, default=30, help="TensorBoard SummaryWriter flush interval (secs).")
    parser.add_argument(
        "--tb_log_every_steps",
        type=int,
        default=50,
        help="Step-level TensorBoard logging frequency during training (0 disables).",
    )
    parser.add_argument(
        "--tb_cm_every_epochs",
        type=int,
        default=5,
        help="Log confusion matrix figures every N epochs (0 disables).",
    )
    parser.add_argument(
        "--tb_worst_k",
        type=int,
        default=10,
        help="How many worst classes (by recall) to report as text/scalars.",
    )
    parser.add_argument(
        "--tb_pr_every_epochs",
        type=int,
        default=0,
        help="Log PR curves every N epochs for worst_k classes (0 disables).",
    )
    parser.add_argument(
        "--tb_hist_every_epochs",
        type=int,
        default=10,
        help="Log weight/gradient/confidence histograms every N epochs (0 disables).",
    )
    parser.add_argument(
        "--tb_profile_first_epoch",
        action="store_true",
        default=False,
        help="Log very light profiling scalars for the first epoch only.",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="If set, ignore any existing last checkpoint and train from scratch.",
    )
    parser.add_argument(
        "--force_resume",
        action="store_true",
        help="If set, resume even if checkpoint class mapping differs (not recommended).",
    )
    parser.add_argument(
        "--quiet_cpp_logs",
        action="store_true",
        default=True,
        help="Suppress MediaPipe/TFLite C++ logs (main + DataLoader workers).",
    )
    parser.add_argument(
        "--no_quiet_cpp_logs",
        dest="quiet_cpp_logs",
        action="store_false",
        help="Do not suppress MediaPipe/TFLite C++ logs.",
    )
    parser.add_argument(
        "--confusion_matrix_path",
        type=str,
        default="",
        help="Optional path to save the TEST confusion matrix (e.g., .npy or .csv).",
    )
    args = parser.parse_args()
    args.log_dir = args.log_dir.strip()

    if args.quiet_cpp_logs and not args.debug_mediapipe:
        os.environ["QUIET_CPP_LOGS"] = "1"
        redirect_stderr_to_devnull_forever()
    else:
        os.environ["QUIET_CPP_LOGS"] = "0"
        restore_stderr_forever()

    seed_everything(args.seed)

    print(f"Scanning dataset at {args.data_root}")
    items, class_to_id = scan_dataset(args.data_root)
    num_classes = len(class_to_id)
    id_to_class = [""] * num_classes
    for cname, idx in class_to_id.items():
        if 0 <= int(idx) < num_classes:
            id_to_class[int(idx)] = str(cname)
    print(f"Found {len(items)} videos across {num_classes} classes.")

    train_items, val_items, test_items = stratified_split(items, class_to_id)
    print(f"Split -> train: {len(train_items)}, val: {len(val_items)}, test: {len(test_items)}")

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    class_counts = compute_class_counts(train_items, num_classes)
    class_weights = compute_cb_class_weights(class_counts, beta=args.cb_beta, weight_norm=args.cb_weight_norm)

    global_step = 0
    writer = None
    if args.log_dir and is_main_process():
        writer = SummaryWriter(log_dir=args.log_dir, flush_secs=int(args.tb_flush_secs))
        writer.add_text("config/args", json.dumps(vars(args), indent=2, sort_keys=True), global_step=0)
        writer.add_text(
            "data/info",
            json.dumps(
                {
                    "data_root": args.data_root,
                    "train_size": len(train_items),
                    "val_size": len(val_items),
                    "test_size": len(test_items),
                    "num_classes": num_classes,
                    "target_len": args.target_len,
                    "cache_dir": args.cache_dir,
                    "imbalance_strategy": args.imbalance_strategy,
                    "cb_beta": args.cb_beta,
                    "cb_weight_norm": args.cb_weight_norm,
                    "disable_sampler": args.disable_sampler,
                },
                indent=2,
                sort_keys=True,
            ),
            global_step=0,
        )
        writer.add_histogram("data/train_class_counts", class_counts.to(torch.float32), global_step=0)
        counts_np = class_counts.cpu().numpy()
        writer.add_scalar("data/train_class_counts_min", float(counts_np.min()) if counts_np.size else 0.0, global_step=0)
        writer.add_scalar("data/train_class_counts_median", float(np.median(counts_np)) if counts_np.size else 0.0, 0)
        writer.add_scalar("data/train_class_counts_max", float(counts_np.max()) if counts_np.size else 0.0, global_step=0)
        if class_weights is not None:
            writer.add_histogram("data/class_weights", class_weights.to(torch.float32), global_step=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    holistic_cfg = {
        "model_complexity": args.model_complexity,
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "debug": args.debug_mediapipe,
    }
    if writer is not None:
        writer.add_scalar(
            "cache/train_hit_rate",
            estimate_cache_hit_rate(train_items, args.cache_dir, args.target_len, holistic_cfg),
            global_step=0,
        )
        writer.add_scalar(
            "cache/val_hit_rate",
            estimate_cache_hit_rate(val_items, args.cache_dir, args.target_len, holistic_cfg),
            global_step=0,
        )
        writer.add_scalar(
            "cache/test_hit_rate",
            estimate_cache_hit_rate(test_items, args.cache_dir, args.target_len, holistic_cfg),
            global_step=0,
        )
    # NOTE (important for Windows + num_workers>0):
    # Each DataLoader worker will lazily create its own MediaPipe Holistic instance.
    # If MediaPipe needs to download models (e.g., pose_landmark_heavy.tflite for model_complexity=2),
    # doing it in multiple workers can trigger concurrent downloads and look like the training "hangs"
    # at 0%. Warm up once in the main process before workers spawn.
    if args.num_workers > 0:
        print("Preloading MediaPipe Holistic (one-time; may download models)...")
        _warmup = HolisticExtractor(**holistic_cfg)
        try:
            _warmup.holistic.close()
        except Exception:
            pass
        del _warmup
    shared_holistic = HolisticExtractor(**holistic_cfg) if args.num_workers == 0 else None
    train_ds = SignDataset(
        train_items,
        target_len=args.target_len,
        holistic_extractor=shared_holistic,
        holistic_cfg=holistic_cfg,
        cache_dir=args.cache_dir,
    )
    val_ds = SignDataset(
        val_items,
        target_len=args.target_len,
        holistic_extractor=shared_holistic,
        holistic_cfg=holistic_cfg,
        cache_dir=args.cache_dir,
    )
    test_ds = SignDataset(
        test_items,
        target_len=args.target_len,
        holistic_extractor=shared_holistic,
        holistic_cfg=holistic_cfg,
        cache_dir=args.cache_dir,
    )

    worker_fn = dataloader_worker_init_silence_cpp_logs if args.num_workers > 0 else None

    train_sampler = None
    if not args.disable_sampler and args.imbalance_strategy in ("sampler", "both"):
        train_sampler = make_weighted_sampler(train_items, num_classes, generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_fn,
        multiprocessing_context="spawn",
        persistent_workers=(args.num_workers > 0),
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_fn,
        multiprocessing_context="spawn",
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_fn,
        multiprocessing_context="spawn",
        persistent_workers=(args.num_workers > 0),
    )

    A = build_adjacency()
    model = DualStreamCTRGCN(
        num_classes=num_classes,
        A=A,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    )
    model = model.to(device)

    criterion = make_cb_criterion(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_f1 = -1.0
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "best_model_ctrgcn_cb.pth")
    last_ckpt_path = os.path.join(ckpt_dir, "last_checkpoint_ctrgcn_cb.pth")
    best_epoch = 0
    epochs_no_improve = 0
    start_epoch = 1

    if not args.no_resume and os.path.exists(last_ckpt_path):
        print(f"Found checkpoint: {last_ckpt_path}")
        try:
            ckpt = torch.load(last_ckpt_path, map_location=device)
            ckpt_class_to_id = ckpt.get("class_to_id")
            if ckpt_class_to_id is not None and ckpt_class_to_id != class_to_id and not args.force_resume:
                raise RuntimeError(
                    "Checkpoint class_to_id differs from current dataset scan. "
                    "Use --force_resume to override or set a different --checkpoint_dir."
                )
            model.load_state_dict(ckpt["model"])
            if ckpt.get("optimizer") is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            best_f1 = float(ckpt.get("best_val_macro_f1", best_f1))
            best_epoch = int(ckpt.get("best_epoch", best_epoch))
            epochs_no_improve = int(ckpt.get("epochs_no_improve", epochs_no_improve))
            last_epoch = int(ckpt.get("epoch", 0))
            start_epoch = max(1, last_epoch + 1)
            print(
                f"Resuming from epoch {start_epoch:03d} "
                f"(best_val_macro_f1={best_f1:.3f} at epoch {best_epoch:03d})."
            )
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint ({last_ckpt_path}): {e}")
            print("Starting training from scratch.")
            start_epoch = 1
            best_f1 = -1.0
            best_epoch = 0
            epochs_no_improve = 0

    model.eval()
    with torch.no_grad():
        sanity_batch = next(iter(train_loader))
        morph_s, traj_s, _ = sanity_batch
        morph_s = morph_s.to(device, non_blocking=True)
        traj_s = traj_s.to(device, non_blocking=True)
        logits_s = model(morph_s, traj_s)
        assert logits_s.shape == (morph_s.shape[0], num_classes), (
            f"Sanity check failed: logits shape {tuple(logits_s.shape)}"
        )
    model.train()

    best_val_snapshot = None
    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_cm, train_quality, train_timing, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            num_classes=num_classes,
            epoch=epoch,
            args=args,
            writer=writer,
            global_step=global_step,
        )
        train_metrics = classification_metrics_from_cm(train_cm)
        train_metrics["loss"] = float(train_loss)
        train_metrics["cm"] = train_cm

        pr_every = int(getattr(args, "tb_pr_every_epochs", 0))
        hist_every = int(getattr(args, "tb_hist_every_epochs", 0))
        need_val_outputs = writer is not None and (
            (pr_every > 0 and epoch % pr_every == 0) or (hist_every > 0 and epoch % hist_every == 0)
        )
        val_metrics, val_outputs = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
            stage="Val",
            return_outputs=need_val_outputs,
            return_probs=need_val_outputs,
        )
        val_loss = float(val_metrics["loss"])
        val_acc = float(val_metrics["top1_acc"])
        val_f1 = float(val_metrics["macro_f1"])

        val_worst_ids: List[int] = []
        if writer is not None:
            log_epoch_metrics_to_tensorboard(writer, "train", train_metrics, epoch, id_to_class, args.tb_worst_k)
            val_worst_ids = log_epoch_metrics_to_tensorboard(writer, "val", val_metrics, epoch, id_to_class, args.tb_worst_k)

            writer.add_scalar("Loss/train", float(train_loss), epoch)
            writer.add_scalar("Loss/val", float(val_loss), epoch)
            writer.add_scalar("Metrics/val_acc", float(val_acc), epoch)
            writer.add_scalar("Metrics/val_macro_f1", float(val_f1), epoch)
            writer.add_scalar("LR", float(current_lr), epoch)

            writer.add_scalar("throughput/train_epoch_time_sec", float(train_timing["epoch_time_sec"]), epoch)
            writer.add_scalar("throughput/train_samples_per_sec", float(train_timing["samples_per_sec"]), epoch)
            writer.add_scalar("throughput/train_avg_step_time_sec", float(train_timing["avg_step_time_sec"]), epoch)
            writer.add_scalar("throughput/train_avg_data_time_sec", float(train_timing["avg_data_time_sec"]), epoch)

            writer.add_scalar("data_epoch/train_morph_near_zero_pct", float(train_quality["morph_near_zero_pct"]), epoch)
            writer.add_scalar("data_epoch/train_traj_near_zero_pct", float(train_quality["traj_near_zero_pct"]), epoch)
            writer.add_scalar("data_epoch/train_morph_motion_energy", float(train_quality["morph_motion_energy"]), epoch)
            writer.add_scalar("data_epoch/train_traj_motion_energy", float(train_quality["traj_motion_energy"]), epoch)
            writer.add_scalar("data_epoch/train_nan_count", float(train_quality["nan_count"]), epoch)

            if bool(getattr(args, "tb_profile_first_epoch", False)) and epoch == start_epoch:
                writer.add_scalar("profile/first_epoch/train_epoch_time_sec", float(train_timing["epoch_time_sec"]), epoch)
                writer.add_scalar("profile/first_epoch/train_samples_per_sec", float(train_timing["samples_per_sec"]), epoch)
                writer.add_scalar("profile/first_epoch/train_avg_step_time_sec", float(train_timing["avg_step_time_sec"]), epoch)
                writer.add_scalar("profile/first_epoch/train_avg_data_time_sec", float(train_timing["avg_data_time_sec"]), epoch)

            cm_every = int(getattr(args, "tb_cm_every_epochs", 0))
            if cm_every > 0 and epoch % cm_every == 0:
                fig = plot_confusion_matrix(
                    train_cm,
                    id_to_class,
                    normalize=True,
                    max_classes=num_classes,
                    title=f"Train confusion matrix (epoch {epoch:03d})",
                )
                if fig is not None:
                    writer.add_figure("train/confusion_matrix", fig, epoch)
                    try:
                        import matplotlib.pyplot as plt

                        plt.close(fig)
                    except Exception:
                        pass
                fig = plot_confusion_matrix(
                    val_metrics["cm"],
                    id_to_class,
                    normalize=True,
                    max_classes=num_classes,
                    title=f"Val confusion matrix (epoch {epoch:03d})",
                )
                if fig is not None:
                    writer.add_figure("val/confusion_matrix", fig, epoch)
                    try:
                        import matplotlib.pyplot as plt

                        plt.close(fig)
                    except Exception:
                        pass

            if hist_every > 0 and epoch % hist_every == 0:
                max_numel = 2_000_000
                for name, param in model.named_parameters():
                    if param is None:
                        continue
                    if int(param.numel()) <= max_numel:
                        writer.add_histogram(f"weights/{name}", param.detach().cpu(), epoch)
                    if param.grad is not None and int(param.grad.numel()) <= max_numel:
                        writer.add_histogram(f"grads/{name}", param.grad.detach().cpu(), epoch)
                if val_outputs is not None and "probs" in val_outputs:
                    conf = val_outputs["probs"].max(dim=1).values
                    writer.add_histogram("val/confidence_max", conf, epoch)

            if pr_every > 0 and epoch % pr_every == 0 and val_outputs is not None and "probs" in val_outputs:
                if hasattr(writer, "add_pr_curve") and int(getattr(args, "tb_worst_k", 0)) > 0:
                    labels_v = val_outputs["labels"]
                    probs_v = val_outputs["probs"]
                    for cid in val_worst_ids:
                        cname = id_to_class[cid] if cid < len(id_to_class) else str(cid)
                        y_true = (labels_v == int(cid)).to(torch.int32)
                        p_c = probs_v[:, int(cid)]
                        try:
                            writer.add_pr_curve(f"val/pr_curve/{_sanitize_tb_tag(cname)}", y_true, p_c, epoch)
                        except Exception:
                            pass
        print(
            f"Epoch {epoch:03d}: "
            f"lr={current_lr:.2e} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_macro_f1={val_f1:.3f}"
        )
        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            best_val_snapshot = {
                "loss": float(val_loss),
                "top1_acc": float(val_acc),
                "macro_f1": float(val_f1),
                "weighted_f1": float(val_metrics["weighted_f1"]),
                "balanced_acc": float(val_metrics["balanced_acc"]),
            }
            _atomic_torch_save(
                best_ckpt_path,
                {
                    "model": model.state_dict(),
                    "class_to_id": class_to_id,
                    "epoch": epoch,
                    "best_val_macro_f1": best_f1,
                    "args": vars(args),
                },
            )
            print(f"Saved new best model with val_macro_f1={best_f1:.3f}")
            if writer:
                writer.add_scalar("Metrics/best_val_macro_f1", best_f1, epoch)
        else:
            epochs_no_improve += 1

        if writer is not None:
            writer.add_scalar("early_stop/best_val_macro_f1", float(best_f1), epoch)
            writer.add_scalar("early_stop/epochs_no_improve", float(epochs_no_improve), epoch)
            writer.add_scalar("early_stop/best_epoch", float(best_epoch), epoch)

        if scheduler is not None:
            scheduler.step()

        _atomic_torch_save(
            last_ckpt_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                "class_to_id": class_to_id,
                "epoch": epoch,
                "best_val_macro_f1": best_f1,
                "best_epoch": best_epoch,
                "epochs_no_improve": epochs_no_improve,
                "args": vars(args),
            },
        )

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch:03d} "
                f"(best epoch {best_epoch:03d}, best val_macro_f1={best_f1:.3f})."
            )
            break

    # Final evaluation on test set
    if os.path.exists(best_ckpt_path):
        best = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best["model"])
    need_test_outputs = writer is not None
    test_metrics, test_outputs = evaluate(
        model,
        test_loader,
        criterion,
        device,
        num_classes,
        stage="Test",
        return_outputs=need_test_outputs,
        return_probs=need_test_outputs,
    )
    test_loss = float(test_metrics["loss"])
    test_acc = float(test_metrics["top1_acc"])
    test_f1 = float(test_metrics["macro_f1"])
    per_class_acc = test_metrics["per_class_recall"].tolist()
    test_cm = test_metrics["cm"]
    if writer is not None:
        step = best_epoch if best_epoch > 0 else args.epochs
        try:
            log_epoch_metrics_to_tensorboard(writer, "test", test_metrics, step, id_to_class, args.tb_worst_k)

            fig = plot_confusion_matrix(
                test_cm,
                id_to_class,
                normalize=True,
                max_classes=num_classes,
                title=f"Test confusion matrix (best epoch {step:03d})",
            )
            if fig is not None:
                writer.add_figure("test/confusion_matrix", fig, step)
                try:
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                except Exception:
                    pass

            writer.add_scalar("Loss/test", float(test_loss), step)
            writer.add_scalar("Metrics/test_acc", float(test_acc), step)
            writer.add_scalar("Metrics/test_macro_f1", float(test_f1), step)
            writer.add_histogram("Metrics/test_per_class_acc", torch.tensor(per_class_acc), step)

            if test_outputs is not None and "probs" in test_outputs and len(test_items) > 0:
                os.makedirs(args.log_dir, exist_ok=True)
                csv_path = os.path.join(args.log_dir, "test_predictions.csv")
                paths = [p for p, _ in test_items]
                true_ids = test_outputs["labels"].to(torch.int64).tolist()
                pred_ids = test_outputs["preds"].to(torch.int64).tolist()
                confs = test_outputs["probs"].max(dim=1).values.to(torch.float32).tolist()
                n = min(len(paths), len(true_ids), len(pred_ids), len(confs))
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["video_path", "true_label", "pred_label", "confidence"])
                    for i in range(n):
                        t = int(true_ids[i])
                        p = int(pred_ids[i])
                        t_name = id_to_class[t] if 0 <= t < len(id_to_class) else str(t)
                        p_name = id_to_class[p] if 0 <= p < len(id_to_class) else str(p)
                        w.writerow([paths[i], t_name, p_name, f"{float(confs[i]):.6f}"])

            hparam_dict = {
                "lr": float(args.lr),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "scheduler": str(args.scheduler),
                "weight_decay": float(args.weight_decay),
                "hidden_dim": int(args.hidden_dim),
                "num_blocks": int(args.num_blocks),
                "dropout": float(args.dropout),
                "target_len": int(args.target_len),
                "label_smoothing": float(args.label_smoothing),
                "imbalance_strategy": str(args.imbalance_strategy),
                "model_complexity": int(args.model_complexity),
                "min_detection_confidence": float(args.min_detection_confidence),
                "min_tracking_confidence": float(args.min_tracking_confidence),
                "seed": int(args.seed),
            }
            metric_dict = {
                "best_val_macro_f1": float(best_f1),
                "best_val_epoch": float(best_epoch),
                "test_loss": float(test_loss),
                "test_top1_acc": float(test_acc),
                "test_macro_f1": float(test_f1),
                "test_weighted_f1": float(test_metrics["weighted_f1"]),
                "test_balanced_acc": float(test_metrics["balanced_acc"]),
            }
            if best_val_snapshot is not None:
                metric_dict.update(
                    {
                        "best_val_loss": float(best_val_snapshot.get("loss", 0.0)),
                        "best_val_top1_acc": float(best_val_snapshot.get("top1_acc", 0.0)),
                        "best_val_weighted_f1": float(best_val_snapshot.get("weighted_f1", 0.0)),
                        "best_val_balanced_acc": float(best_val_snapshot.get("balanced_acc", 0.0)),
                    }
                )
            try:
                writer.add_hparams(hparam_dict, metric_dict)
            except Exception:
                pass
        finally:
            try:
                writer.flush()
            except Exception:
                pass
            try:
                writer.close()
            except Exception:
                pass
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.3f} macro_f1={test_f1:.3f}")
    print("Per-class accuracy (ordered by class id):")
    print(per_class_acc)

    if args.confusion_matrix_path:
        out_path = args.confusion_matrix_path
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cm_np = test_cm.cpu().numpy()
        if out_path.lower().endswith(".npy"):
            np.save(out_path, cm_np)
        else:
            np.savetxt(out_path, cm_np, fmt="%d", delimiter=",")
        print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
