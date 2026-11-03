#!/usr/bin/env python3

from __future__ import annotations
import csv, glob, json, os, sys
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
PROJ=os.path.dirname(os.path.dirname(HERE))
from retina.models import build_unet, build_espcn, enhance
from retina.predict import segment
from retina.metrics import image_metrics, segmentation_metrics, bootstrap_mean
from retina.masking import mask_image


DRIVE=os.path.join(HERE,"data","DRIVE_training")
DATA=os.path.join(HERE,"data")
UNET_W=os.path.join(HERE,"weights","baseline_unet_aug_do_0.1_activation_ReLU_weights.best.hdf5")
ESPCN_W=os.path.join(HERE,"weights/espcn_best.weights.h5")
OUT=os.path.join(HERE,"reproduccion_paper"); os.makedirs(OUT,exist_ok=True)
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
PUB={"Set5":{"bicubic":30.39,"ours":37.92},"Set14":{"bicubic":27.53,"ours":37.46}}

def load_models():
    unet=build_unet(32,0.1); unet.load_weights(UNET_W)
    espcn=build_espcn(3); espcn.load_weights(ESPCN_W)
    return unet, espcn

def sr_table(espcn):
    rows={}
    for ds in ("Set5","Set14"):
        bic=[]; esp=[]
        for hrp in sorted(glob.glob(f"{DATA}/{ds}/*_HR.png")):
            hr=np.asarray(Image.open(hrp).convert("RGB"),np.float32)/255.
            H,W=hr.shape[:2]; h3,w3=H-H%3,W-W%3; hr=hr[:h3,:w3]
            lr=np.asarray(Image.fromarray(np.uint8(hr*255)).resize((w3//3,h3//3),Image.BICUBIC),np.float32)/255.
            up=np.asarray(Image.fromarray(np.uint8(lr*255)).resize((w3,h3),Image.BICUBIC),np.float32)/255.
            sr=enhance(lr, espcn, 3, mode="y", iterations=1).astype(np.float32)
            sr=np.asarray(Image.fromarray(np.uint8(np.clip(sr*255,0,255))).resize((w3,h3),Image.BICUBIC),np.float32)/255. if sr.shape[:2]!=(h3,w3) else sr
            bic.append(image_metrics(hr,up,crop=3,channel="y")["psnr_db"])
            esp.append(image_metrics(hr,sr,crop=3,channel="y")["psnr_db"])
        rows[ds]=dict(bicubic=float(np.mean(bic)), espcn=float(np.mean(esp)),
                      pub_bicubic=PUB[ds]["bicubic"], pub_ours=PUB[ds]["ours"], n=len(bic))
    return rows

def seg_table(unet):
    mets={k:[] for k in ("dice","iou","precision","recall","specificity","roc_auc")}
    for ip in sorted(glob.glob(f"{DRIVE}/images/*.tif")):
        stem=os.path.basename(ip).split("_")[0]
        gtp=f"{DRIVE}/1st_manual/{stem}_manual1.gif"; fovp=f"{DRIVE}/mask/{stem}_training_mask.gif"
        if not os.path.exists(gtp): continue
        rgb=np.asarray(Image.open(ip).convert("RGB"),np.float32)/255.
        gt=np.asarray(Image.open(gtp).convert("L"))>127
        fov=(np.asarray(Image.open(fovp).convert("L"))>127) if os.path.exists(fovp) else None
        prob=segment(unet,rgb)
        m=segmentation_metrics(gt,prob,fov=fov,threshold=0.5)
        for k in mets: mets[k].append(m[k])
    return {k:dict(mean=float(np.mean(v)), **{x:round(bootstrap_mean(v)[x],4) for x in ("ci_low","ci_high")}, n=len(v)) for k,v in mets.items()}

def panel_figure(unet, espcn, subject, img_path, fig_no):
    rgb=np.asarray(Image.open(img_path).convert("RGB"),np.float32)/255.
    H,W=rgb.shape[:2]; h3,w3=H-H%3,W-W%3; base=rgb[:h3,:w3]
    lr=np.asarray(Image.fromarray(np.uint8(base*255)).resize((w3//3,h3//3),Image.BICUBIC),np.float32)/255.
    sr=enhance(lr, espcn, 3, mode="y", iterations=1).astype(np.float32)
    prob=segment(unet, base)
    masked=mask_image(np.uint8(base*255), threshold=100)["morphed"]
    fig,ax=plt.subplots(1,4,figsize=(14,3.6))
    for a,(im,t) in zip(ax,[(base,"(a) Code"),(np.clip(sr,0,1),"(b) super-resuelta ESPCN"),
                            (prob,"(c) segmentación U-Net"),(masked,"(d) masking")]):
        a.imshow(im, cmap="gray" if im.ndim==2 else None); a.set_title(t,fontsize=9); a.axis("off")
    fig.suptitle(f"Figura {fig_no} — {subject}",fontsize=11)
    fig.tight_layout()
    p=f"{OUT}/fig{fig_no}_{subject}_retina2024.png"; fig.savefig(p,dpi=150,bbox_inches="tight"); plt.close(fig)
    return p

def main():
    unet, espcn = load_models()
    print("U-Net + ESPCN", flush=True)
    sr=sr_table(espcn); seg=seg_table(unet)
    json.dump({"sr":sr,"seg":{k:v for k,v in seg.items()}}, open(f"{OUT}/info_paper.json","w"), indent=2)
    print("\n=== TABLA SR (PSNR dB, canal Y) ===")
    for ds in ("Set5","Set14"):
        r=sr[ds]; print(f"  {ds}: bicúbico reprod={r['bicubic']:.2f} (pub {r['pub_bicubic']}) | ESPCN 2024={r['espcn']:.2f} | Ours pub={r['pub_ours']}")
    print("\n=== TABLA SEGMENTACIÓN (DRIVE 21-40, n=20) ===")
    for k,v in seg.items(): print(f"  {k:12s} {v['mean']:.4f}  IC[{v['ci_low']},{v['ci_high']}]")
    figs=[panel_figure(unet,espcn,"retina", sorted(glob.glob(f"{DRIVE}/images/*.tif"))[0], 4),
          panel_figure(unet,espcn,"baboon", f"{DATA}/Set14/img_001_SRF_3_HR.png", 5)]
    print("\nfiguras:", [os.path.basename(f) for f in figs])
    print(f"-> {OUT}/")

if __name__=="__main__": main()
