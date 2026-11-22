"""Manifiesto con fuentes/sesiones independientes y revisión vigente."""
from collections import Counter,defaultdict
import hashlib
from pathlib import Path
import os
import numpy as np
from torch.utils.data import Dataset
import torch
from .common import canonical,digest,read_csv,read_json,write_csv,save_json
from .paths import resolve
from .config import contract_hash
from .preprocessing import base_array,difference_hash,read_image,tensor_image
from .inference import preprocess
from .acquisition import catalog,load_reviews
from .augmentation import augment_moderate

SPLITS=('train','val','test')

def validate_rows(rows,labels,minimum=1,minimum_unknown=1):
    if minimum<1 or minimum_unknown<1:raise ValueError('Los mínimos deben ser positivos')
    if not rows:raise ValueError('Manifiesto vacío')
    unknown=len(labels)-1;groups=defaultdict(set);fingerprints={};sources={};paths=set()
    for row in rows:
        split=row['split'];label=int(row['class_id'])
        if split not in SPLITS or label not in range(len(labels)):raise ValueError('Clase/split inválidos')
        if label!=unknown and row['identity']!=labels[label]:raise ValueError('Etiqueta/identidad incompatible')
        key=('unknown',row['identity']) if label==unknown else ('session',row['identity'],row['session'])
        groups[key].add(split)
        if row['path'] in paths:raise ValueError('Ruta duplicada')
        paths.add(row['path'])
        for field in ('sha256','pixel_sha256'):
            value=row[field]
            if value in fingerprints:raise ValueError('Duplicado exacto; revisar sin reasignar test')
            fingerprints[value]=split
        old=sources.setdefault(row['source_id'],(split,row['identity']))
        if old!=(split,row['identity']):raise ValueError('Fuente compartida entre splits/identidades')
    if any(len(v)>1 for v in groups.values()):raise ValueError('Fuga de identidad desconocida/sesión')
    for split in SPLITS:
        for label in range(len(labels)):
            subset=[r for r in rows if r['split']==split and int(r['class_id'])==label]
            if len(subset)<minimum:raise ValueError(f'Mínimo insuficiente en {split}/clase{label}: {len(subset)}/{minimum}')
        if len({r['identity'] for r in rows if r['split']==split and int(r['class_id'])==unknown})<minimum_unknown:raise ValueError('Faltan identidades desconocidas disjuntas')
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a['split']!=b['split'] and (int(a['dhash'])^int(b['dhash'])).bit_count()<=3:raise ValueError('Casi duplicado entre splits; revisar sin mover test')


def prepare(cfg,base):
    base=Path(base);root=resolve(cfg['data_root'],base);manifest=resolve(cfg['manifest'],base)
    if manifest.exists() or manifest.with_suffix('.json').exists():raise FileExistsError('Conservar manifiesto; elegir otra ruta/version')
    labels=list(cfg['authorized'])+['Desconocido']
    if len(set(labels))!=len(labels) or len(labels)<2:raise ValueError('Identidades autorizadas únicas y no vacías')
    mapping=cfg['session_splits']
    if set(mapping.values())!=set(SPLITS):raise ValueError('Configurar sesiones para train/val/test')
    decisions=load_reviews(root);rows=[]
    for row in catalog(root):
        event=decisions.get(row['sample_id'])
        if not event or event['status']!='accepted':continue
        if event['image_sha256']!=row['sha256']:raise ValueError('Revisión desactualizada')
        image=read_image(root/row['path']);preprocess(image)
        if row['kind']=='authorized':
            if row['identity'] not in cfg['authorized']:raise ValueError('Identidad capturada no está en authorized; actualizar clases y reentrenar')
            if row['session'] not in mapping:raise ValueError('Sesión no definida en el plan')
            split=mapping[row['session']];label=labels.index(row['identity'])
        else:
            if row['identity'] in cfg['authorized']:raise ValueError('Identidad autorizada en desconocidos')
            split=row['split'];label=len(labels)-1
        rows.append(dict(row,path=Path(os.path.relpath(root/row['path'],manifest.parent)).as_posix(),split=split,class_id=label,
            pixel_sha256=hashlib.sha256(base_array(image,'none').tobytes()).hexdigest(),dhash=str(difference_hash(image))))
    validate_rows(rows,labels,cfg['minimum_per_identity'],cfg['minimum_unknown_identities'])
    # Unknown identities also need the stated minimum, not merely an aggregate unknown class.
    counts=Counter((r['split'],r['identity']) for r in rows)
    if any(n<cfg['minimum_per_identity'] for n in counts.values()):raise ValueError('Una identidad no alcanza el mínimo aprobado')
    write_csv(manifest,rows)
    save_json(manifest.with_suffix('.json'),dict(labels=labels,manifest_sha256=digest(manifest),contract_sha256=contract_hash(),
        data_root_relative=Path(os.path.relpath(root,manifest.parent)).as_posix(),minimum=cfg['minimum_per_identity'],minimum_unknown=cfg['minimum_unknown_identities'],
        session_splits=mapping,counts={f'{split}/{identity}':n for (split,identity),n in counts.items()}))
    return manifest


def audit_manifest(path):
    path=Path(path).resolve();meta=read_json(path.with_suffix('.json'));rows=read_csv(path)
    if digest(path)!=meta['manifest_sha256'] or meta['contract_sha256']!=contract_hash():raise ValueError('Manifiesto/contrato modificado')
    root=resolve(meta['data_root_relative'],path.parent);decisions=load_reviews(root)
    # Re-check the original acquisition inventory and human decisions before training/evaluation.
    originals={r['sample_id']:r for r in catalog(root)}
    for row in rows:
        original=originals.get(row['sample_id']);event=decisions.get(row['sample_id'])
        if not original or not event or event['status']!='accepted' or event['image_sha256']!=row['sha256']:raise ValueError('Aprobación ausente/revocada')
        if resolve(row['path'],path.parent)!=(root/original['path']).resolve():raise ValueError('Ruta de imagen alterada')
        for key in ('sha256','identity','session','kind','source_id','source_frame','contract_sha256'):
            if str(row[key])!=str(original[key]):raise ValueError('Procedencia alterada: '+key)
        expected_split=meta['session_splits'][row['session']] if row['kind']=='authorized' else original['split']
        if row['split']!=expected_split:raise ValueError('Partición alterada')
        expected_label=meta['labels'].index(row['identity']) if row['kind']=='authorized' else len(meta['labels'])-1
        if int(row['class_id'])!=expected_label:raise ValueError('Clase alterada')
        image=read_image(resolve(row['path'],path.parent));preprocess(image)
        if row['pixel_sha256']!=hashlib.sha256(base_array(image,'none').tobytes()).hexdigest() or int(row['dhash'])!=difference_hash(image):raise ValueError('Fingerprint alterado')
    validate_rows(rows,meta['labels'],meta['minimum'],meta['minimum_unknown'])
    return rows,meta


class FaceDataset(Dataset):
    def __init__(self,rows,manifest,split,augmentation=None,seed=42):
        self.rows=[r for r in rows if r['split']==split];self.base=Path(manifest).resolve().parent
        self.augmentation=augmentation if split=='train' else None;self.generator=torch.Generator().manual_seed(seed+100)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];image=read_image(resolve(row['path'],self.base));tensor=preprocess(image)
        if self.augmentation:
            if self.augmentation['name']=='moderate':tensor=augment_moderate(tensor,self.generator,self.augmentation)
            elif self.augmentation['name']=='original':tensor=tensor_image(image,'median',96,augment=True,generator=self.generator)
            elif self.augmentation['name']!='none':raise ValueError('Aumento no reconocido')
        return tensor,int(row['class_id'])
