"""Train one source-only transfer. Author: Moh'd Khier Al Kfari."""
from __future__ import annotations
import json,pickle
from pathlib import Path
import numpy as np
from .experiments import Transfer,resolve_group
from .metrics import evaluate
from .model import SelfHARCNN
from .utils import SEED,seed_everything,write_json
ROOT=Path(__file__).resolve().parents[2]
def _load(family,name):
    dataset,group=resolve_group(name); folder=ROOT/"data"/"prepared"/family/dataset
    if group: folder/=group
    if not (folder/"features.pkl").is_file(): raise FileNotFoundError(f"Prepared features missing: {folder/'features.pkl'}")
    with (folder/"features.pkl").open("rb") as h: features=pickle.load(h)
    with (folder/"evaluation_labels.pkl").open("rb") as h:
        label_payload=pickle.load(h)
    labels=label_payload.get("labels", label_payload)
    meta={}; mp=ROOT/"data"/"prepared"/family/"metadata"/f"{dataset}{('_'+group) if group else ''}.json"
    if mp.is_file(): meta=json.loads(mp.read_text(encoding="utf-8"))
    meta.setdefault("label_map", label_payload.get("label_map", {}))
    return features,labels,meta
def _arrays(obj):
    return {s:np.concatenate([np.asarray(v) for v in part.values()]) if isinstance(part,dict) else np.asarray(part) for s,part in obj.items()}
def _label_map(meta): return meta.get("label_map") or meta.get("metadata",{}).get("activities") or {}
def run_transfer(transfer:Transfer,epochs=30,batch_size=32,lr=3e-4,weight_decay=1e-4,seed=SEED,gpu=None):
    seed_everything(seed); import torch
    device=torch.device(f"cuda:{gpu}" if gpu is not None and torch.cuda.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    sf,sl,sm=_load(transfer.family,transfer.source); tf,tl,tm=_load(transfer.family,transfer.target)
    sf,sl,tf,tl=map(_arrays,(sf,sl,tf,tl)); lm,tmx=_label_map(sm),_label_map(tm)
    if lm and tmx and set(lm)!=set(tmx): raise ValueError(f"Label maps differ for {transfer.name}: {lm} vs {tmx}")
    x,y=sf["train"].astype(np.float32),sl["train"].astype(np.int64)
    if not len(x): raise ValueError(f"Source training set is empty: {transfer.name}")
    model=SelfHARCNN((x.shape[1],x.shape[2]),max(len(lm or tmx),int(y.max()+1))).to(device); opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay); loss_fn=torch.nn.CrossEntropyLoss(); gen=torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for order in torch.randperm(len(x),generator=gen).split(batch_size):
            xb=torch.as_tensor(x[order.numpy()],dtype=torch.float32,device=device); yb=torch.as_tensor(y[order.numpy()],dtype=torch.long,device=device); opt.zero_grad(); loss_fn(model(xb),yb).backward(); opt.step()
    out=ROOT/"models"/transfer.family/transfer.name; (out/"predictions").mkdir(parents=True,exist_ok=True); metrics={}
    for prefix,features,labels in (("source",sf,sl),("target",tf,tl)):
        for split in ("train","val","test"):
            result=evaluate(model,features[split].astype(np.float32),labels[split],device); metrics[f"{prefix}_{split}"]={k:v for k,v in result.items() if k!="predictions"}; np.savez_compressed(out/"predictions"/f"{prefix}_{split}.npz",y_true=labels[split],y_pred=result["predictions"])
    torch.save(model.state_dict(),out/"model_last.pt"); write_json(out/"config.json",{"transfer":transfer.name,"family":transfer.family,"source":transfer.source,"target":transfer.target,"seed":seed,"epochs":epochs,"batch_size":batch_size,"learning_rate":lr,"weight_decay":weight_decay,"device":str(device),"label_map":lm or tmx,"training_shuffle":False}); write_json(out/"metrics.json",{"final_epoch":metrics}); return out
