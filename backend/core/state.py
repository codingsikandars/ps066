"""
Backend configuration and model state loader with dataset denormalization support.
"""

from pathlib import Path
import yaml
import numpy as np
import torch
import sys

# Ensure repo root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.oceanembed_net import build_oceanembed_model
from src.data.dataset import OceanDataset
from src.data.preprocess import STANDARD_DEPTHS, TARGET_LATS, TARGET_LONS

CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "oceanembed_best.pt"
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"


class OceanEmbedState:
    def __init__(self):
        self.config = {}
        self.model = None
        self.dataset = None
        self.test_inputs = None  # raw inputs (N, 7, 101, 241)
        self.test_targets = None  # raw targets in degC (N, 15, 101, 241)
        self.mask = None
        self.grid_metadata = {}
        self.argo_observations = []
        self.evaluation_report = {}
        self.training_history = {}
        self.lats = TARGET_LATS
        self.lons = TARGET_LONS
        self.depths = STANDARD_DEPTHS
        self.stats = None
        self.is_ready = False

    def load_all(self):
        if self.is_ready:
            return

        print("[OceanEmbed Backend] Loading configuration...")
        with open(CONFIG_PATH, "r") as f:
            self.config = yaml.safe_load(f)

        print("[OceanEmbed Backend] Loading land-ocean mask and grid metadata...")
        mask_path = DATA_DIR / "ocean_land_mask.npy"
        if mask_path.exists():
            self.mask = np.load(mask_path)
        else:
            self.mask = np.ones((len(self.lats), len(self.lons)), dtype=bool)

        grid_meta_path = DATA_DIR / "grid_metadata.npz"
        if grid_meta_path.exists():
            gm = np.load(grid_meta_path)
            self.grid_metadata = {k: gm[k].tolist() if hasattr(gm[k], "tolist") else gm[k] for k in gm.files}

        print("[OceanEmbed Backend] Loading ARGO in-situ observations...")
        argo_path = DATA_DIR / "argo_observations.npy"
        if argo_path.exists():
            raw_argo = np.load(argo_path, allow_pickle=True)
            self.argo_observations = [dict(item) for item in raw_argo]
            print(f"[OceanEmbed Backend] Loaded {len(self.argo_observations)} ARGO float profiles")

        print("[OceanEmbed Backend] Loading evaluation report...")
        report_path = OUTPUTS_DIR / "evaluation_report.json"
        if report_path.exists():
            import json
            with open(report_path, "r") as f:
                self.evaluation_report = json.load(f)

        history_path = ROOT_DIR / "checkpoints" / "training_history.json"
        if history_path.exists():
            import json
            with open(history_path, "r") as f:
                self.training_history = json.load(f)

        print("[OceanEmbed Backend] Loading PyTorch model checkpoint...")
        self.model = build_oceanembed_model(self.config)
        self.model.eval()
        if CHECKPOINT_PATH.exists():
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            self.stats = ckpt.get("stats", None)
            print("[OceanEmbed Backend] Model checkpoint and statistics loaded successfully!")

        print("[OceanEmbed Backend] Initializing OceanDataset for test partition...")
        test_path = DATA_DIR / "test_data.npz"
        if test_path.exists():
            raw_td = np.load(test_path)
            self.test_inputs = raw_td["inputs"]  # (N, 7, 101, 241)
            self.test_targets = raw_td["targets"]  # (N, 15, 101, 241)

            self.dataset = OceanDataset(
                data_path=test_path,
                use_coordconv=self.config["model"].get("use_coordconv", True),
                time_window=self.config.get("temporal", {}).get("time_window", 1),
                normalize=True,
                stats=self.stats,
                fill_missing=True,
            )
            print(f"[OceanEmbed Backend] Test data loaded: {len(self.dataset)} daily snapshots")

        self.is_ready = True
        print("[OceanEmbed Backend] All systems ready.")


state = OceanEmbedState()
