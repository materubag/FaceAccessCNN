"""Entrenamiento desde cero; mejor val_loss, AMP y early stopping del prototipo."""
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .common import seed_everything,digest,save_json,write_csv
from .config import contract_hash,input_contract
from .dataset import audit_manifest,FaceDataset
from .model import FaceCNN
from .inference import FORMAT,weights_digest


def train(cfg,manifest,output,smoke=False,device='auto'):
    output=Path(output)
    if output.exists():raise FileExistsError('Conservar ejecución existente: cambiar output')
    rows,meta=audit_manifest(manifest)
    if meta['labels']!=list(cfg['authorized'])+['Desconocido']:raise ValueError('Clases distintas; preparar de nuevo y reentrenar')
    if meta['session_splits']!=cfg['session_splits']:raise ValueError('Plan de sesiones distinto; preparar manifiesto nuevo')
    seed_everything(cfg['seed']);device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
    model_config=dict(cfg['model'],num_classes=len(meta['labels']));model=FaceCNN(**model_config).to(device)
    settings=cfg['training'];generator=torch.Generator().manual_seed(cfg['seed'])
    loaders={split:DataLoader(FaceDataset(rows,manifest,split,cfg['augmentation'] if split=='train' else None,cfg['seed']),
        batch_size=settings['batch_size'],shuffle=split=='train',num_workers=0,generator=generator) for split in ('train','val')}
    counts=np.bincount([int(r['class_id']) for r in rows if r['split']=='train'],minlength=len(meta['labels']))
    weights=torch.tensor(counts.sum()/(len(counts)*counts),device=device,dtype=torch.float32)
    criterion=nn.CrossEntropyLoss(weight=weights);optimizer=torch.optim.Adam(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    amp=settings['amp'] and device.type=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=amp)
    output.mkdir(parents=True);best=float('inf');stale=0;history=[];started=time.perf_counter()
    for epoch in range(1,(1 if smoke else settings['epochs'])+1):
        start=time.perf_counter();record=dict(epoch=epoch)
        for split in ('train','val'):
            training=split=='train';model.train(training);loss_sum=correct=total=0;weight_sum=0.
            for x,y in loaders[split]:
                x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training),torch.autocast(device_type=device.type,enabled=amp):logits=model(x);loss=criterion(logits,y)
                if training:scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
                batch_weight=weights[y].sum().item();weight_sum+=batch_weight;loss_sum+=loss.item()*batch_weight
                correct+=(logits.argmax(1)==y).sum().item();total+=len(y)
            record[split+'_loss']=loss_sum/weight_sum;record[split+'_accuracy']=correct/total
        record.update(lr=settings['lr'],epoch_seconds=time.perf_counter()-start);history.append(record)
        if record['val_loss']<best:
            best=record['val_loss'];stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(dict(format=FORMAT,model_config=model_config,weights=state,weights_sha256=weights_digest(state),labels=meta['labels'],
                input_contract=input_contract(),contract_sha256=contract_hash(),manifest_sha256=digest(manifest),smoke_only=smoke,experimental=True,model_id='trained',
                epoch=epoch,best_val_loss=best,seed=cfg['seed'],training=settings,augmentation=cfg['augmentation'],decision=cfg['decision'],class_weights=weights.cpu().tolist()),output/'best.pt')
        else:stale+=1
        write_csv(output/'history.csv',history);save_json(output/'history.json',history);print(record,flush=True)
        if stale>=settings['patience']:break
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,key in zip(axes,('loss','accuracy')):
        for split in ('train','val'):ax.plot([r['epoch'] for r in history],[r[split+'_'+key] for r in history],label=split)
        ax.set_title(key);ax.set_xlabel('Epoch');ax.legend()
    fig.tight_layout();fig.savefig(output/'learning_curves.png');plt.close(fig)
    save_json(output/'summary.json',dict(epochs=len(history),best_epoch=min(history,key=lambda r:r['val_loss'])['epoch'],best_val_loss=best,
        seconds=time.perf_counter()-started,device=str(device),amp=amp,parameters=sum(p.numel() for p in model.parameters())))
    return output/'best.pt'


@torch.inference_mode()
def collect(model,dataset,device,batch_size=32):
    model.eval();probabilities=[];labels=[];elapsed=0.;total=0
    # Warm-up on synthetic zeros, not on reserved images.
    for _ in range(3):model(torch.zeros(1,1,96,96,device=device))
    if device.type=='cuda':torch.cuda.synchronize()
    for x,y in DataLoader(dataset,batch_size=batch_size,shuffle=False):
        x=x.to(device)
        if device.type=='cuda':torch.cuda.synchronize()
        started=time.perf_counter();logits=model(x)
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed+=time.perf_counter()-started;total+=len(y)
        probabilities.extend(logits.softmax(1).cpu().numpy());labels.extend(y.numpy())
    return np.asarray(labels),np.asarray(probabilities),elapsed/total
