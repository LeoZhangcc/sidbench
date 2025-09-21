
import argparse, os, subprocess, sys, csv, tempfile, shutil

DEFAULTS = {
    "DIMD": {"ckpt": "/content/sidbench/weights/dimd/corvi22_latent_model.pth", "size": 256, "batch": 32},
    "Rine": {"ckpt": "/content/sidbench/weights/rine/model_ldm_trainable.pth", "size": 224, "batch": 64},  # CLIP→224
    "DeFake": {
        "ckpt": "/content/sidbench/weights/defake/clip_linear.pth",
        "defakeClipEncoderPath": "/content/sidbench/weights/defake/finetune_clip.pt",
        "defakeBlipPath": "/content/sidbench/weights/defake/model_base_capfilt_large.pth",
        "size": 224, "batch": 16
    },
    "PatchCraft": {"ckpt": "/content/sidbench/weights/rptc/RPTC.pth", "size": 256, "batch": 64},
    "CNNDetect": {"ckpt": "/content/sidbench/weights/cnndetect/blur_jpg_prob0.1.pth", "size": 256, "batch": 64},
}

MODEL_NAME_MAP = {"PatchCraft": "RPTC"}

def run_one(model, data_path, csv_dir, threads, overrides):
    cfg = DEFAULTS[model].copy()
    # overlay allowed：--Rineckpt / --Rinesize / --DeFakesize / ...
    for k, v in list(overrides.items()):
        if v and k.lower().startswith(model.lower()):
            key = k[len(model):]; key = key[0].lower() + key[1:]
            cfg[key] = v
    real_name = MODEL_NAME_MAP.get(model, model)
    out_csv = os.path.join(csv_dir, f"results_{model.lower()}.csv")

    cmd = [sys.executable, "test.py",
           "--dataPath", data_path, "--modelName", real_name, "--ckpt", cfg["ckpt"],
           "--predictionsFile", out_csv, "--loadSize", str(cfg["size"]),
           "--cropSize", str(cfg["size"]), "--batchSize", str(cfg["batch"]),
           "--numThreads", str(threads)]
    if model == "DeFake":
        cmd += ["--defakeClipEncoderPath", cfg["defakeClipEncoderPath"],
                "--defakeBlipPath", cfg["defakeBlipPath"]]
    print(">> Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd="/content/sidbench")
    if r.returncode != 0:
        raise SystemExit(f"[ERROR] {model} failed with code {r.returncode}")
    return out_csv

def merge_csv(csv_paths, out_csv):
    # Read in the CSV files of each model and merge them: scores, labels (>0.5 = 1), average scores, majority votes
    by_img, order = {}, []
    for p in csv_paths:
        m = os.path.basename(p).split("_", 1)[1].split(".")[0]     # dimd / rine / defake / patchcraft / cnndetect
        m = {"rptc":"PatchCraft"}.get(m, m.capitalize())           
        order.append(m)
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                img = row["Image Path"]; s = float(row["Prediction"])
                by_img.setdefault(img, {})[m] = s

    header = ["Image Path"] + sum(([m, f"{m}_label"] for m in order), [])
    header += ["avg_score", "vote_label"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for img, rec in by_img.items():
            row, votes, scores = [img], [], []
            for m in order:
                s = rec.get(m, float("nan")); scores.append(s)
                lab = 1 if s > 0.5 else 0
                row += [s, lab]; votes.append(lab)
            valid = [s for s in scores if s == s]
            avg = sum(valid)/len(valid) if valid else float("nan")
            vote = 1 if sum(votes) > len(votes)/2 else 0
            row += [avg, vote]; w.writerow(row)

if __name__ == "__main__":
    ap = argparse.ArgumentParser("SIDBench multi-model runner (only combined CSV)")
    ap.add_argument("--dataPath", required=True)
    ap.add_argument("--outDir", default="/content/sidbench/multi_out")
    ap.add_argument("--numThreads", type=int, default=2)
    
    for m in ["DIMD","Rine","DeFake","PatchCraft","CNNDetect"]:
        ap.add_argument(f"--skip{m}", action="store_true")
    # Cover ckpt/size/additional paths
    ap.add_argument("--DIMDckpt"); ap.add_argument("--Rineckpt")
    ap.add_argument("--DeFakeckpt"); ap.add_argument("--DeFakedefakeClipEncoderPath"); ap.add_argument("--DeFakedefakeBlipPath")
    ap.add_argument("--PatchCraftckpt"); ap.add_argument("--CNNDetectckpt")
    ap.add_argument("--DIMDsize", type=int); ap.add_argument("--Rinesize", type=int)
    ap.add_argument("--DeFakesize", type=int); ap.add_argument("--PatchCraftsize", type=int); ap.add_argument("--CNNDetectsize", type=int)
    # If you want to keep the single model CSV file, then enable this switch to override the ckpt/size/extra path.
    ap.add_argument("--keepPerModel", action="store_true")

    args = ap.parse_args()

    # Single model CSV directory: By default, uses the temporary directory and deletes it after merging; if keepPerModel is set, uses outDir
    tmpdir = None
    if args.keepPerModel:
        csv_dir = args.outDir
        os.makedirs(csv_dir, exist_ok=True)
    else:
        tmpdir = tempfile.mkdtemp(prefix="sid_multi_")
        csv_dir = tmpdir

    models = [m for m in ["DIMD","Rine","DeFake","PatchCraft","CNNDetect"] if not getattr(args, f"skip{m}")]
    overrides = {k:v for k,v in vars(args).items() if v and not k.startswith("skip")
                 and k not in ["dataPath","outDir","numThreads","keepPerModel"]}

    csvs = [run_one(m, args.dataPath, csv_dir, args.numThreads, overrides) for m in models]

    merged = os.path.join(args.outDir, "results_combined.csv")
    merge_csv(csvs, merged)
    # Automatically clean the CSV of a single model
    if tmpdir: shutil.rmtree(tmpdir, ignore_errors=True)
    print("Done. Combined CSV:", merged)

