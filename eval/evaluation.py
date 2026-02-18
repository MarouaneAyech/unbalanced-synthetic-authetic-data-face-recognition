import logging
import os
import glob
import sys

# Ajouter le répertoire parent (racine du projet) au path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from utils.utils_callbacks import CallBackVerification
from utils.utils_logging import init_logging

from config.config import config as cfg

from backbones.iresnet import iresnet100, iresnet50, iresnet34

if __name__ == "__main__":
    gpu_id = 0
    log_root = logging.getLogger()

    # Charger le backbone selon la config
    if cfg.network == "iresnet100":
        print("Loading iresnet100")
        backbone = iresnet100(num_features=cfg.embedding_size).to(f"cuda:{gpu_id}")
    elif cfg.network == "iresnet50":
        print("Loading iresnet50")
        backbone = iresnet50(num_features=cfg.embedding_size).to(f"cuda:{gpu_id}")
    elif cfg.network == "iresnet34":
        print("Loading iresnet34")
        backbone = iresnet34(num_features=cfg.embedding_size).to(f"cuda:{gpu_id}")
    else:
        print(f"Network {cfg.network} not supported!")
        backbone = None
        exit()

    # Trouver automatiquement le dernier checkpoint backbone.pth
    # Chercher dans les sous-dossiers aussi (recursive search)
    checkpoint_pattern = os.path.join(cfg.output, "**", "*backbone.pth")
    checkpoints = glob.glob(checkpoint_pattern, recursive=True)
    
    if checkpoints:
        # Prendre le dernier checkpoint (le plus récent)
        weights_path = max(checkpoints, key=os.path.getctime)
        print(f"Loading weights from: {weights_path}")
        backbone.load_state_dict(torch.load(weights_path, map_location=f"cuda:{gpu_id}"))
    else:
        print(f"No checkpoint found in {cfg.output}")
        print(f"Searched pattern: {checkpoint_pattern}")
        print(f"Make sure you have trained a model first!")
        print(f"Expected structure: {cfg.output}/exp1/iresnet34_CosFace_WF_0K_/*backbone.pth")
        exit()

    # Initialiser logging
    init_logging(log_root, 0, cfg.output, logfile="evaluation.log")
    
    # Callback de vérification
    callback_verification = CallBackVerification(1, 0, cfg.val_targets, cfg.val_root)

    # Wrapper DataParallel
    model = torch.nn.DataParallel(backbone, device_ids=[gpu_id])
    
    # Lancer l'évaluation
    print("Starting evaluation...")
    callback_verification(1, model)
    print("Evaluation completed!")

