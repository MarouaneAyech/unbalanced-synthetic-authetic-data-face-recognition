"""
Final Evaluation Script - Évalue le modèle entraîné sur TOUS les benchmarks
avec gestion correcte de la mémoire GPU pour éviter OOM.

Usage:
    python eval/final_evaluation.py
"""

import logging
import os
import glob
import sys
import gc
import torch

# Ajouter le répertoire parent (racine du projet) au path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from eval import verification
from config.config import config as cfg
from backbones.iresnet import iresnet100, iresnet50, iresnet34


def evaluate_single_benchmark(backbone, benchmark_name, cfg, gpu_id=0):
    """
    Évalue le modèle sur UN SEUL benchmark et libère la mémoire après.
    
    Args:
        backbone: Le modèle chargé
        benchmark_name: Nom du benchmark (ex: "lfw")
        cfg: Configuration
        gpu_id: GPU ID à utiliser
    
    Returns:
        dict: Résultats d'évaluation pour ce benchmark
    """
    benchmark_path = os.path.join(cfg.val_root, benchmark_name + ".bin")
    
    if not os.path.exists(benchmark_path):
        print(f"⚠️  Benchmark not found: {benchmark_path}")
        return {
            "benchmark": benchmark_name,
            "status": "NOT_FOUND",
            "accuracy": None,
            "accuracy_std": None,
            "val": None,
            "val_std": None,
            "far": None,
            "xnorm": None
        }
    
    try:
        print(f"\n{'='*60}")
        print(f"Evaluating: {benchmark_name.upper()}")
        print(f"{'='*60}")
        
        # Charger le benchmark UNE FOIS
        print(f"Loading benchmark: {benchmark_path}")
        data_set = verification.load_bin(benchmark_path, image_size=(112, 112))
        
        # Évaluer sur ce benchmark
        backbone.eval()
        with torch.no_grad():
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                data_set, backbone, batch_size=10, nfolds=10, device=f'cuda:{gpu_id}')
        
        print(f"✓ {benchmark_name.upper()}: Accuracy={acc2:.5f}±{std2:.5f}, XNorm={xnorm:.4f}")
        
        # IMPORTANT: Libérer la mémoire du benchmark
        del data_set, embeddings_list
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "benchmark": benchmark_name,
            "status": "SUCCESS",
            "accuracy": acc2,
            "accuracy_std": std2,
            "val": None,  # À calculer si nécessaire
            "val_std": None,
            "far": None,
            "xnorm": xnorm
        }
    
    except Exception as e:
        print(f"❌ Error evaluating {benchmark_name}: {str(e)}")
        # Libérer la mémoire en cas d'erreur
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "benchmark": benchmark_name,
            "status": "ERROR",
            "error": str(e),
            "accuracy": None,
            "accuracy_std": None,
            "val": None,
            "val_std": None,
            "far": None,
            "xnorm": None
        }


def main():
    """Main evaluation function"""
    
    gpu_id = 0
    print(f"Using GPU: {gpu_id}")
    print(f"Device available: {torch.cuda.is_available()}")
    
    # Charger le backbone
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
        print(f"Unknown network: {cfg.network}")
        exit()
    
    # Chercher le checkpoint
    checkpoint_pattern = os.path.join(cfg.output, "**", "*backbone.pth")
    checkpoints = list(glob.glob(checkpoint_pattern, recursive=True))
    
    if not checkpoints:
        print(f"❌ No checkpoint found!")
        print(f"Searched pattern: {checkpoint_pattern}")
        print(f"Make sure you have trained a model first!")
        print(f"Expected structure: {cfg.output}/exp1/iresnet34_CosFace_WF_0K_/*backbone.pth")
        exit()
    
    # Charger le dernier checkpoint (ou le meilleur)
    latest_checkpoint = sorted(checkpoints)[-1]
    print(f"\nLoading checkpoint: {latest_checkpoint}")
    state_dict = torch.load(latest_checkpoint, map_location=f"cuda:{gpu_id}")
    backbone.load_state_dict(state_dict)
    backbone.eval()
    
    # Déterminer les benchmarks à évaluer
    test_targets = getattr(cfg, 'test_targets', cfg.val_targets)
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Network: {cfg.network}")
    print(f"Embedding size: {cfg.embedding_size}")
    print(f"Checkpoint: {latest_checkpoint}")
    print(f"Test targets: {test_targets}")
    print(f"{'='*60}\n")
    
    # Évaluer UN benchmark à la fois pour éviter OOM
    results = []
    for benchmark_name in test_targets:
        result = evaluate_single_benchmark(backbone, benchmark_name, cfg, gpu_id)
        results.append(result)
    
    # Afficher le résumé final
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Benchmark':<15} {'Status':<10} {'Accuracy':<15} {'XNorm':<10}")
    print(f"{'-'*60}")
    
    for result in results:
        benchmark = result['benchmark']
        status = result['status']
        
        if status == "SUCCESS":
            accuracy = f"{result['accuracy']:.5f}±{result['accuracy_std']:.5f}"
            xnorm = f"{result['xnorm']:.4f}"
            print(f"{benchmark:<15} {status:<10} {accuracy:<15} {xnorm:<10}")
        elif status == "ERROR":
            print(f"{benchmark:<15} {status:<10} {'N/A':<15} {'N/A':<10}")
            if 'error' in result:
                print(f"  └─ Error: {result['error'][:50]}")
        else:
            print(f"{benchmark:<15} {status:<10} {'N/A':<15} {'N/A':<10}")
    
    print(f"{'='*60}\n")
    
    # Sauvegarder les résultats
    import json
    results_file = os.path.join(cfg.output, "final_evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
