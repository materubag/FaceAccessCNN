from pathlib import Path
import numpy as np
from .common import digest,save_json,write_csv
from .dataset import audit_manifest,FaceDataset
from .inference import Predictor
from .metrics import access_metrics,calibrate
from .training import collect


def evaluate(checkpoint,manifest,output,split='val',calibration=None,target_far=None,margin=None,confirm_frozen=False,device='auto'):
    output=Path(output);checkpoint=Path(checkpoint)
    if output.exists():raise FileExistsError('Conservar evaluación existente')
    if split not in {'val','test'}:raise ValueError('Evaluar solo val o test')
    if split=='val' and calibration is not None:raise ValueError('Validación calibra; no pasar calibración de otro modelo')
    if split=='test' and (calibration is None or not confirm_frozen):raise ValueError('Test requiere calibración val y --confirm-frozen')
    if split=='test' and (target_far is not None or margin is not None):raise ValueError('Test usa calibración congelada; no admite overrides de umbral/margen')
    marker=checkpoint.parent/'test_started.json'
    if split=='test' and marker.exists():raise FileExistsError('Test ya iniciado/evaluado; no repetir para ajustar')
    rows,meta=audit_manifest(manifest);predictor=Predictor(checkpoint,calibration if split=='test' else None,device)
    if split=='val' and predictor.checkpoint['model_id'] in {'r02','r03b'}:raise ValueError('Modelos de referencia congelados: no recalibrar')
    if predictor.checkpoint['manifest_sha256']!=digest(manifest) or predictor.labels!=meta['labels']:raise ValueError('Manifiesto no corresponde al entrenamiento')
    if split=='test' and predictor.checkpoint['smoke_only']:raise ValueError('Smoke no puede realizar evaluación final')
    if split=='test':save_json(marker,dict(checkpoint_sha256=digest(checkpoint),manifest_sha256=digest(manifest),started=True),exclusive=True)
    dataset=FaceDataset(rows,manifest,split);y,probability,seconds=collect(predictor.model,dataset,predictor.device)
    policy=predictor.checkpoint.get('decision',dict(target_far=.01,margin=.15))
    target_far=policy['target_far'] if target_far is None else target_far
    margin=policy['margin'] if margin is None else margin
    cal=predictor.calibration if split=='test' else calibrate(y,probability,target_far,margin)
    if split=='val':cal.update(checkpoint_sha256=digest(checkpoint),manifest_sha256=digest(manifest),calibrated_split='val')
    metrics=access_metrics(y,probability,cal['threshold'],cal['margin'],meta['labels'])
    metrics.update(split=split,samples=len(y),checkpoint_sha256=digest(checkpoint),calibration=cal,mean_forward_seconds=seconds,
        timing_scope='Forward amortizado en batches; excluye detector/preprocesamiento y no es FPS webcam')
    output.mkdir(parents=True);save_json(output/'metrics.json',metrics)
    if split=='val':save_json(output/'calibration.json',cal)
    write_csv(output/'predictions.csv',[dict(sample_id=row['sample_id'],true=int(label),**{f'p{i}':float(v) for i,v in enumerate(p)}) for row,label,p in zip(dataset.rows,y,probability)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,('confusion_matrix','access_confusion_matrix'),('Clasificación','Aceptación')):
        ConfusionMatrixDisplay(np.asarray(metrics[key]),display_labels=meta['labels']).plot(ax=ax,colorbar=False,xticks_rotation=30);ax.set_title(title)
    fig.tight_layout();fig.savefig(output/'confusion.png');plt.close(fig)
    return metrics
