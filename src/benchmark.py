import csv
import json
import sys
from argparse import Namespace

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from .fetch import CONFIG, prepare


def disparity(path):
    # Middlebury stores pixel disparity directly. The PFM header sign encodes endianness.
    with path.open("rb") as f:
        if f.readline().strip() != b"Pf":
            raise ValueError(f"Expected grayscale PFM: {path}")
        w,h = map(int, f.readline().split())
        little_endian = float(f.readline()) < 0
        samples = np.fromfile(f, dtype="<f4" if little_endian else ">f4")
    return np.flipud(samples.reshape(h,w)).astype("f",copy=True)


def load_scene(folder):
    def read_image(name, flag):
        value = cv2.imread(str(folder/name),flag)
        if value is None:
            raise ValueError(folder/name)
        return value
    source = read_image("im0.png", cv2.IMREAD_COLOR)
    width = CONFIG["width"]
    scale = width / source.shape[1]
    size = (width, round(source.shape[0]*scale))
    left, right = [cv2.resize(a, size, interpolation=cv2.INTER_AREA) for a in (source, read_image("im1.png",1))]
    truth, rtruth = [cv2.resize(disparity(folder/name),size,interpolation=cv2.INTER_NEAREST)*scale
                      for name in ("disp0.pfm","disp1.pfm")]
    return left, right, truth, rtruth


def sgbm(left, right):
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5,
        P1=8*3*25, P2=32*3*25, uniquenessRatio=8, speckleWindowSize=25,
        speckleRange=2, disp12MaxDiff=2)
    d = matcher.compute(left,right).astype(np.float32)/16
    d[d < 0] = np.nan
    return d


def raft_estimator(cache, device):
    sys.path.insert(0,str((cache/"raft").resolve()))
    from core.raft_stereo import RAFTStereo
    from core.utils.utils import InputPadder
    args = Namespace(hidden_dims=[128]*3, context_norm="batch", n_downsample=2, n_gru_layers=3,
                     corr_implementation="reg", corr_levels=4, corr_radius=4, shared_backbone=False,
                     slow_fast_gru=False, mixed_precision=False)
    model = RAFTStereo(args)
    sd = torch.load(cache/"raftstereo-sceneflow.pth",map_location="cpu",weights_only=True)
    model.load_state_dict({k.removeprefix("module."):v for k,v in sd.items()})
    model.to(device).eval()
    @torch.inference_mode()
    def predict(a,b):
        pair = [torch.from_numpy(cv2.cvtColor(i,cv2.COLOR_BGR2RGB)).permute(2,0,1)[None].to(device).float() for i in (a,b)]
        padder = InputPadder(pair[0].shape, divis_by=32)
        _, flow = model(*padder.pad(*pair), iters=CONFIG["iterations"], test_mode=True)
        return -padder.unpad(flow)[0,0].cpu().numpy()
    return predict


def lr_error(d, right):
    y,x = np.indices(d.shape)
    valid = np.isfinite(d) & (d >= 0)
    coord = np.rint(x-np.where(valid,d,0)).astype(int)
    inside = valid & (coord>=0) & (coord<d.shape[1])
    sampled = right[y,np.clip(coord,0,d.shape[1]-1)]
    loss = np.full_like(d,np.inf)
    ok = inside & np.isfinite(sampled) & (sampled>=0)
    loss[ok] = np.abs(d[ok]-sampled[ok])
    return loss


def masks(left, truth, right_truth):
    g = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY).astype("f")
    mean = cv2.blur(g,(7,7))
    std = np.sqrt(np.maximum(cv2.blur(g*g,(7,7))-mean*mean,0))
    periodic = np.full_like(g,np.inf)
    for period in (4,8,12,16):
        repeat = cv2.blur(np.abs(g-np.roll(g,period,axis=1)),(7,7))
        repeat[:,:period+3] = np.inf
        periodic = np.minimum(periodic,repeat)
    residual = lr_error(truth,right_truth)
    y,x = np.indices(g.shape)
    off_image = x < truth
    return {"all":np.ones(g.shape,bool), "weak_texture": std<CONFIG["low_texture_std"],
            "repeated_texture_proxy":(std>=CONFIG["low_texture_std"]) & (periodic<CONFIG["repetition_mae"]),
            "occlusion_proxy":off_image | (np.isfinite(residual) & (residual>1.0))}


def coverage_plot(records, out):
    scenes = sorted({r[0] for r in records})
    width, height = 255, 210
    chart = Image.new("RGB", (width*len(scenes), height+25), "white")
    draw = ImageDraw.Draw(chart)
    draw.text((10,4), "EPE in pixels versus retained fraction of all valid ground-truth pixels", fill="black")
    colors = {"sgbm":"#b25c26", "raft":"#176b83"}
    for panel,scene in enumerate(scenes):
        left, right = panel*width+38, (panel+1)*width-16
        top, bottom = 55, 185
        draw.text((left,29),scene,fill="black")
        rows=[r for r in records if r[0]==scene and r[2]=="all" and r[7]!=""]
        maxerr=max(1.,float(np.ceil(max(r[7] for r in rows)*2)/2))
        for i in range(3):
            y=bottom-i*(bottom-top)/2
            draw.line((left,y,right,y),fill="#dedede")
            draw.text((left-32,y-5),f"{maxerr*i/2:.1f}",fill="#333333")
            x=left+i*(right-left)/2
            draw.text((x-8,bottom+3),f"{i/2:.1f}",fill="#333333")
        for idx,(method,color) in enumerate(colors.items()):
            seq=sorted((r[6],r[7]) for r in rows if r[1]==method)
            pts=[(left+x*(right-left),bottom-y/maxerr*(bottom-top)) for x,y in seq]
            draw.line(pts,fill=color,width=2)
            for x,y in pts: draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
            draw.text((left+idx*70,207),method.upper(),fill=color)
    chart.save(out/"coverage.png")


def evaluate(out):
    records = []
    for file in sorted(out.glob("*.npz")):
        with np.load(file) as p:
            for method in ("sgbm","raft"):
                pred = p[method]
                err = np.abs(pred-p["truth"])
                valid_gt = np.isfinite(p["truth"]) & (p["truth"]>=0)
                for region in ("all","weak_texture","repeated_texture_proxy","occlusion_proxy"):
                    region_gt = valid_gt & p["mask_"+region]
                    total = int(region_gt.sum())
                    for t in [None,*CONFIG["lr_thresholds"]]:
                        keep = region_gt & np.isfinite(pred) & (pred>=0)
                        if t is not None:
                            keep &= p[method+"_lr"]<=t
                        n = int(keep.sum())
                        records.append([file.stem,method,region,"none" if t is None else t,total,n,
                              n/total if total else "",float(err[keep].mean()) if n else "",
                              float((err[keep]>1).mean()) if n else ""])
    if not records:
        raise FileNotFoundError("Generate predictions with reproduce first")
    with (out/"metrics.csv").open("w",newline="") as f:
        w = csv.writer(f); w.writerow(["scene","method","region","max_lr_px","valid_gt","retained","coverage","epe_px","bad1"]); w.writerows(records)
    coverage_plot(records,out)
    print(out/"metrics.csv")


def reproduce(cache, out, device, names):
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    checksums = prepare(cache,names)
    out.mkdir(exist_ok=True,parents=True)
    (out/"assets.json").write_text(json.dumps(checksums,indent=2))
    raft = raft_estimator(cache,device)
    for scene in names:
        left,right,truth,rtruth = load_scene(cache/"stereodata"/scene)
        values = {"truth":truth, **{"mask_"+k:v for k,v in masks(left,truth,rtruth).items()}}
        for method,fn in (("sgbm",sgbm),("raft",raft)):
            pred = fn(left,right)
            right_pred = np.fliplr(fn(np.fliplr(right).copy(),np.fliplr(left).copy()))
            values[method] = pred
            values[method+"_lr"] = lr_error(pred,right_pred)
        np.savez_compressed(out/(scene+".npz"),**values)
        print(scene,flush=True)
        if scene == names[0]:
            panels = [truth,values["sgbm"],values["raft"],np.abs(values["raft"]-truth)]
            h,w = truth.shape
            image = Image.new("RGB",(w*2,(h+21)*2),"white")
            for idx, (a,label) in enumerate(zip(panels,["Ground truth, 0-64px","SGBM, 0-64px","RAFT, 0-64px","RAFT abs. error, 0-5px"])):
                finite = np.isfinite(a)
                scaled = np.clip(np.where(finite,a,0)/(5 if idx==3 else 64)*255,0,255).astype("uint8")
                color = cv2.applyColorMap(scaled,cv2.COLORMAP_VIRIDIS)[:,:,::-1].copy()
                color[~finite] = 0
                x,y = (idx%2)*w,(idx//2)*(h+21)
                image.paste(Image.fromarray(color),(x,y+21))
                ImageDraw.Draw(image).text((x+4,y+3),label, fill="black")
            image.save(out/"comparison.png")
    evaluate(out)
