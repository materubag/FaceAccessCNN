import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .inference import decisions


def access_metrics(y, probability, threshold, margin, labels):
    y=np.asarray(y);raw=np.asarray(probability).argmax(1);post=decisions(probability,threshold,margin)
    unknown=y==len(labels)-1;known=~unknown;indices=list(range(len(labels)))
    if not unknown.any() or not known.any():raise ValueError('Se requieren autorizados y desconocidos')
    return dict(accuracy=float(accuracy_score(y,raw)),macro_f1=float(f1_score(y,raw,labels=indices,average='macro',zero_division=0)),
        per_class=classification_report(y,raw,labels=indices,target_names=labels,output_dict=True,zero_division=0),
        confusion_matrix=confusion_matrix(y,raw,labels=indices).tolist(),access_confusion_matrix=confusion_matrix(y,post,labels=indices).tolist(),
        access_accuracy=float(accuracy_score(y,post)),far=float(np.mean(post[unknown]!=len(labels)-1)),frr=float(np.mean(post[known]==len(labels)-1)),
        correct_authorized_acceptance=float(np.mean(post[known]==y[known])),authorized_wrong_identity_rate=float(np.mean((post[known]!=len(labels)-1)&(post[known]!=y[known]))),
        unknown_count=int(unknown.sum()),authorized_count=int(known.sum()))


def calibrate(y,probability,target_far=.01,margin=.15):
    y=np.asarray(y);probability=np.asarray(probability);unknown=probability.shape[1]-1
    if not 0<=target_far<=1 or not 0<=margin<=1:raise ValueError('Objetivo FAR/margen fuera de rango')
    if not (y==unknown).any() or not (y!=unknown).any():raise ValueError('Validación requiere ambas poblaciones')
    best=None
    for threshold in np.unique(np.r_[0.,probability.max(1),1.000001]):
        pred=decisions(probability,float(threshold),margin);far=float(np.mean(pred[y==unknown]!=unknown))
        correct=float(np.mean(pred[y!=unknown]==y[y!=unknown]))
        if far<=target_far:
            key=(correct,float(threshold))
            if best is None or key>best[0]:best=(key,float(threshold),far)
    return dict(threshold=best[1],margin=margin,target_far=target_far,validation_far=best[2],validation_correct_authorized_rate=best[0][0],
        validation_unknown_count=int((y==unknown).sum()),warning='Softmax y margen son heurísticas; FAR empírico no garantiza seguridad ni detección de vida.')
