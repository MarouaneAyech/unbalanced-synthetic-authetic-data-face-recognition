from easydict import EasyDict as edict

config = edict()
config.auth_dataset = "WF"
config.synt_dataset = "DC"  # training dataset
config.embedding_size = 128  # embedding size of model (réduit pour GPU faible)
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 8  # batch size per GPU (réduit pour GPU faible, était 32)
config.lr = 0.1
config.output = "output/"  # train model output folder
config.global_step = 0  # step to resume
config.s = 64.0
config.m = 0.35
config.std = 0.05

config.loss = "CosFace"  # Option : ElasticArcFace, ArcFace, ElasticCosFace, CosFace, MLLoss, ElasticArcFacePlus, ElasticCosFacePlus, AdaFace

if config.loss == "ElasticArcFacePlus":
    config.s = 64.0
    config.m = 0.50
    config.std = 0.0175
elif config.loss == "ElasticArcFace":
    config.s = 64.0
    config.m = 0.50
    config.std = 0.05
if config.loss == "ElasticCosFacePlus":
    config.s = 64.0
    config.m = 0.35
    config.std = 0.02
elif config.loss == "ElasticCosFace":
    config.s = 64.0
    config.m = 0.35
    config.std = 0.05
elif config.loss == "AdaFace":
    config.s = 64.0
    config.m = 0.4
elif config.loss == "ArcFace":
    config.s = 64.0
    config.m = 0.5

config.auth_dict = {
    "WF": r"/content/drive/MyDrive/research_face_recog/datasets/webface",
    # "M2-S": "/data/Authentic/faces_emore_10k/images",
    # "BUPT": "/data/Authentic/faces_emore_10k/images",
    # "BUPT_bal": "/data/Authentic/faces_emore_10k/images",
}

config.synt_dict = {
    "DC": r"/content/drive/MyDrive/research_face_recog/datasets/dcface",
    # "GC_bal": "/data/Synthetic/GC_bal/images",
    # "DC": "/data/Synthetic/dcface_0.5m_oversample_xid/images",
    # "DC_bal": "/data/Synthetic/DC_bal/images",
    # "IDF": "/data/Synthetic/Idifface/images",
    # "IDF_bal": "/data/Synthetic/IDF_bal/images",
}

config.synthetic_root = config.synt_dict[config.synt_dataset]


config.val_root = r"/content/drive/MyDrive/research_face_recog/datasets/benchmarks"
config.network = "iresnet34" # [ iresnet34 | iresnet50 | iresnet100 ]
config.SE = False  # SEModule

config.rec = config.auth_dict[config.auth_dataset]
config.num_epoch = 2  # Réduit à 2 pour test rapide (était 5)
config.warmup_epoch = -1
config.val_targets = ["lfw"]  # Un seul benchmark pour test rapide
config.test_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]

def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
        [m for m in [3,4,5] if m - 1 <= epoch])

config.lr_func = lr_step_func
