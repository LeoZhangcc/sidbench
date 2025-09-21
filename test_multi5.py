%%writefile /content/sidbench/test_multi5.py
import argparse, os, subprocess, sys, csv

# ===== 默认权重与输入尺寸（按你的实际路径改） =====
DEFAULTS = {
    "DIMD": {"ckpt": "/content/sidbench/weights/dimd/corvi22_latent_model.pth", "size": 256, "batch": 32},
    "Rine": {"ckpt": "/content/sidbench/weights/rine/model_ldm_trainable.pth", "size": 224, "batch": 64},  # CLIP家族→224
    "DeFake": {
        "ckpt": "/content/sidbench/weights/defake/clip_linear.pth",
        "defakeClipEncoderPath": "/content/sidbench/weights/defake/finetune_clip.pt",
        "defakeBlipPath": "/content/sidbench/weights/defake/model_base_capfilt_large.pth",
        "size": 256, "batch": 16
    },
    "PatchCraft": {"ckpt": "/content/sidbench/weights/rptc/RPTC.pth", "size": 256, "batch": 64},
    "CNNDetect": {"ckpt": "/content/sidbench/weights/cnndetect/blur_jpg_prob0.1.pth", "size": 256, "batch": 64},
}

def run_one(model, data_path, out_dir, threads, overrides):
    cfg = DEFAULTS[model].copy()
    # 允许从命令行覆盖各模型路径：如 --Rineckpt /path/x.pth 或 --DeFakedefakeBlipPath /path/y.pth
    for k, v in list(overrides.items()):
        if v and k.lower().startswith(model.lower()):
            key = k[len(model):]
            key = key[0].lower() + key[1:]
            cfg[key] = v

    out_csv = os.path.join(out_dir, f"results_{model.lower()}.csv")
    cmd = [
        sys.executable, "test.py",
        "--dataPath", data_path,
        "--modelName", model,
        "--ckpt", cfg["ckpt"],
        "--predictionsFile", out_csv,
        "--loadSize", str(cfg["size"]),
        "--cropSize", str(cfg["size"]),
        "--batchSize", str(cfg["batch"]),
        "--numThreads", str(threads),
    ]
    if model == "DeFake":
        cmd += ["--defakeClipEncoderPath", cfg["defakeClipEncoderPath"], "--defakeBlipPath", cfg["defakeBlipPath"]]

    print(">> Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd="/content/sidbench")
    if r.returncode != 0:
        raise SystemExit(f"[ERROR] {model} failed with code {r.returncode}")
    return out_csv

def merge_csv(csv_paths, out_csv):
    # 合并：每图各模型分数 + 标签(>0.5为1) + 平均分 + 多数投票
    by_img = {}; order = []
    for p in csv_paths:
        m = os.path.basename(p).split("_", 1)[1].split(".")[0]  # e.g., dimd
        m = {"rptc":"PatchCraft"}.get(m, m.capitalize())
        order.append(m)
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                img = row["Image Path"]; s = float(row["Prediction"])
                by_img.setdefault(img, {})[m] = s

    header = ["Image Path"] + sum(([m, f"{m}_label"] for m in order), [])
    header += ["avg_score", "vote_label"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for img, rec in by_img.items():
            row, votes, scores = [img], [], []
            for m in order:
                s = rec.get(m, float("nan")); scores.append(s)
                lab = 1 if (s > 0.5) else 0   # 与 test.py 阈值一致
                row += [s, lab]; votes.append(lab)
            valid = [s for s in scores if s == s]
            avg = sum(valid)/len(valid) if valid else float("nan")
            vote = 1 if sum(votes) > len(votes)/2 else 0
            row += [avg, vote]; w.writerow(row)

if __name__ == "__main__":
    ap = argparse.ArgumentParser("SIDBench multi-model runner (5 models)")
    ap.add_argument("--dataPath", required=True)
    ap.add_argument("--outDir", default="/content/sidbench/multi_out")
    ap.add_argument("--numThreads", type=int, default=2)
    # 可跳过
    for m in ["DIMD","Rine","DeFake","PatchCraft","CNNDetect"]:
        ap.add_argument(f"--skip{m}", action="store_true")
    # 覆盖路径
    ap.add_argument("--DIMDckpt"); ap.add_argument("--Rineckpt")
    ap.add_argument("--DeFakeckpt"); ap.add_argument("--DeFakedefakeClipEncoderPath"); ap.add_argument("--DeFakedefakeBlipPath")
    ap.add_argument("--PatchCraftckpt"); ap.add_argument("--CNNDetectckpt")
    args = ap.parse_args()

    os.makedirs(args.outDir, exist_ok=True)
    models = [m for m in ["DIMD","Rine","DeFake","PatchCraft","CNNDetect"] if not getattr(args, f"skip{m}")]
    overrides = {k:v for k,v in vars(args).items() if v and not k.startswith("skip") and k not in ["dataPath","outDir","numThreads"]}

    csvs = [run_one(m, args.dataPath, args.outDir, args.numThreads, overrides) for m in models]
    merged = os.path.join(args.outDir, "results_combined.csv"); merge_csv(csvs, merged)
    print("Done. Combined CSV:", merged)
