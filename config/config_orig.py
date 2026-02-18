from easydict import EasyDict as edict

config = edict()
config.auth_dataset = "WF"
config.synt_dataset = "GC"  # training dataset
config.embedding_size = 512  # embedding size of model
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128  # batch size per GPU
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
    "WF": "/data/Authentic/casia_training/images",
    "M2-S": "/data/Authentic/faces_emore_10k/images",
    "BUPT": "/data/Authentic/faces_emore_10k/images",
    "BUPT_bal": "/data/Authentic/faces_emore_10k/images",
}

config.synt_dict = {
    "GC": "/data/Synthetic/GAN_Control_class_images/images",
    "GC_bal": "/data/Synthetic/GC_bal/images",
    "DC": "/data/Synthetic/dcface_0.5m_oversample_xid/images",
    "DC_bal": "/data/Synthetic/DC_bal/images",
    "IDF": "/data/Synthetic/Idifface/images",
    "IDF_bal": "/data/Synthetic/IDF_bal/images",
}

config.synthetic_root = config.synt_dict[config.synt_dataset]


config.val_root = "/data/FR_Benchmark"
config.network = "iresnet34" # [ iresnet34 | iresnet50 | iresnet100 ]
config.SE = False  # SEModule


config.rec = config.auth_dict[config.auth_dataset]
config.num_epoch = 40  # [22, 30, 35]
config.warmup_epoch = -1
config.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw", "African_test", "Asian_test", "Caucasian_test", "Indian_test"]


def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
        [m for m in [22, 30, 36] if m - 1 <= epoch])


config.lr_func = lr_step_func
