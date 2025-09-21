import os
import argparse
import csv
from tqdm import tqdm
import torch

from dataset.dataset import RecursiveImageDataset
from dataset.process import processing
from dataset import patch_collate

from models import get_model
from options import TestOptions
from utils.util import setup_device


# 你当前权重目录下可用的模型（基于你贴的 weights 结构）
ENABLED_MODELS = [
    "CNNDetect",
    "DIMD",
    "Dire",
    "FreqDetect",
    "Fusing",
    "GramNet",
    "LGrad",
    "NPR",
    "UnivFD",
    "PatchCraft",
    "DeFake",
    "Rine",
]

# 各模型主权重路径（根据你截图列出的文件名）
CKPT_PATHS = {
    "CNNDetect": "weights/cnndetect/blur_jpg_prob0.1.pth",
    "DIMD": "weights/dimd/corvi22_latent_model.pth",
    "Dire": "weights/dire/lsun_adm.pth",
    "FreqDetect": "weights/freqdetect/DCTAnalysis.pth",
    "Fusing": "weights/fusing/PSM.pth",
    "GramNet": "weights/gramnet/Gram.pth",
    "LGrad": "weights/lgrad/LGrad-1class-Trainon-Progan_horse.pth",
    "NPR": "weights/npr/NPR.pth",
    "UnivFD": "weights/univfd/fc_weights.pth",
    "PatchCraft": "weights/rptc/RPTC.pth",
    "DeFake": "weights/defake/clip_linear.pth",
    "Rine": "weights/rine/model_ldm_trainable.pth",
}

# 特殊依赖（README 里要求的附加权重/文件）
EXTRA_ARGS = {
    "DeFake": {
        "defakeClipEncoderPath": "weights/defake/finetune_clip.pt",
        "defakeBlipPath": "weights/defake/model_base_capfilt_large.pth",
    },
    "FreqDetect": {
        "dctMean": "weights/freqdetect/dct_mean",  # 注意需解压到目录
        "dctVar": "weights/freqdetect/dct_var",
    },
    "LGrad": {
        "LGradGenerativeModelPath": "weights/preprocessing/karras2019stylegan-bedrooms-256x256_discriminator.pth",
    },
    "Dire": {
        "DireGenerativeModelPath": "weights/preprocessing/lsun_bedroom.pt",
    },
}

# CLIP 家族需要 224 输入
CLIP_MODELS = {"UnivFD", "Rine"}


def run_one_model(opt, model_name, results_dir):
    # 设置当前模型与 ckpt
    opt.modelName = model_name
    opt.ckpt = CKPT_PATHS[model_name]

    # CLIP 系列：若用户未指定，则强制 224
    if model_name in CLIP_MODELS and (not hasattr(opt, "resizeSize") or opt.resizeSize in (0, None)):
        opt.resizeSize = 224  # README 提示：CLIP 需要 224x224

    # 特殊参数（若存在）
    extras = EXTRA_ARGS.get(model_name, {})
    for k, v in extras.items():
        setattr(opt, k, v)

    # 结果路径
    os.makedirs(results_dir, exist_ok=True)
    predictions_file = os.path.join(results_dir, f"{model_name}_predictions.csv")

    print(f"\n==== Running {model_name} ====")
    print(f"ckpt: {opt.ckpt}")
    print(f"dataPath: {opt.dataPath}")
    print(f"predictionsFile: {predictions_file}")

    # 设备
    device = setup_device(opt.gpus)

    # 模型
    model = get_model(opt)

    # Fusing 的自定义 collate
    collate_fn = patch_collate if model_name == "Fusing" else None

    # 数据
    dataset = RecursiveImageDataset(data_path=opt.dataPath, opt=opt, process_fn=processing)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=opt.numThreads,
        collate_fn=collate_fn,
    )

    # 推理
    collected = []
    with tqdm(total=len(dataset)) as pbar:
        for img, label, img_path in loader:
            if isinstance(img, list):
                img = [i.to(device) if isinstance(i, torch.Tensor) else i for i in img]
                preds = model.predict(*img)
            else:
                img = img.to(device)
                preds = model.predict(img)

            # 按 0.5 阈值二分类（与单模型 test.py 保持一致）
            bin_labels = [1 if p > 0.5 else 0 for p in preds]

            for pth, pred, bl in zip(img_path, preds, bin_labels):
                collected.append((model_name, pth, float(pred), int(bl)))

            pbar.update(len(bin_labels))

    # 写 CSV（追加或覆盖都行，这里覆盖为每模型独立文件）
    with open(predictions_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ModelName", "Image Path", "Prediction", "Label"])
        writer.writerows(collected)


def main():
    # 复用 TestOptions（保证和 test.py 同一套参数）
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = TestOptions().initialize(parser)

    # 额外加一个输出目录参数（不影响原有）
    parser.add_argument("--resultsDir", type=str, default="results_all", help="Directory to save per-model CSVs")
    parser.add_argument("--models", type=str, nargs="*", default=ENABLED_MODELS,
                        help="Subset of models to run. If omitted, run all available.")
    opt = parser.parse_args()

    # 跑指定（或全部）模型
    for m in opt.models:
        if m not in CKPT_PATHS:
            print(f"[Skip] {m}: no ckpt path configured.")
            continue
        # 基础文件存在性检查（尽量友好提示，不强制退出）
        ckpt_ok = os.path.exists(CKPT_PATHS[m])
        extras = EXTRA_ARGS.get(m, {})
        extras_ok = all(os.path.exists(p) for p in extras.values())
        if not ckpt_ok:
            print(f"[Warn] {m}: ckpt not found at {CKPT_PATHS[m]} (skipping).")
            continue
        if not extras_ok:
            print(f"[Warn] {m}: some extra files not found {extras} (attempting to run anyway).")

        run_one_model(opt, m, opt.resultsDir)


if __name__ == "__main__":
    main()
