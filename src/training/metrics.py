"""Evaluation metrics. Author: Moh'd Khier Al Kfari."""
from __future__ import annotations
import numpy as np
def evaluate(model,x,y,device):
    import torch
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score
    model.eval(); pred=[]
    with torch.no_grad():
        for i in range(0,len(x),512): pred.append(model(torch.as_tensor(x[i:i+512],dtype=torch.float32,device=device)).argmax(1).cpu().numpy())
    pred=np.concatenate(pred) if pred else np.empty(0,dtype=int); y=np.asarray(y,dtype=int)
    return {"accuracy":float(accuracy_score(y,pred)) if len(y) else 0.,"macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)) if len(y) else 0.,"balanced_accuracy":float(balanced_accuracy_score(y,pred)) if len(y) else 0.,"confusion_matrix":confusion_matrix(y,pred).tolist() if len(y) else [],"predictions":pred}
