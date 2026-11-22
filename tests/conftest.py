import copy
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from face_access.common import digest,canonical,save_json
from face_access.config import contract_hash
from face_access.acquisition import decide
from face_access.dataset import prepare


@pytest.fixture
def dataset_case(tmp_path):
    base=tmp_path;root=base/'data/captures';cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/train.yaml').read_text(encoding='utf-8'))
    cfg.update(authorized=['alice','bob'],minimum_per_identity=1,minimum_unknown_identities=2)
    cfg['model'].update(channels=[2,4,8],hidden=8);cfg['training'].update(epochs=1,batch_size=2,amp=False)
    cfg['augmentation']['name']='none'
    rng=np.random.default_rng(300);records=[]
    for split,session in [('train','sesion_01'),('val','sesion_02'),('test','sesion_03')]:
        for person,kind in [('alice','authorized'),('bob','authorized'),(f'u_{split}_1','unknown'),(f'u_{split}_2','unknown')]:
            branch=root/kind/person/session/'sample' if kind=='authorized' else root/kind/split/person/session/'sample'
            branch.mkdir(parents=True);path=branch/'crop.jpg';cv2.imwrite(str(path),rng.integers(0,256,(96,96,3),dtype=np.uint8))
            row=dict(sample_id=canonical([person,session,digest(path)])[:24],path=path.relative_to(root).as_posix(),sha256=digest(path),identity=person,
                session=session,kind=kind,split=None if kind=='authorized' else split,source_id='synthetic:'+person+session,source_frame=0,source_seconds=0.,
                automatic_status='candidate',score=.99,contract_sha256=contract_hash())
            save_json(branch/'inventory.json',[row]);save_json(branch/'run.json',dict(complete=True,consent_confirmed=True,independence_confirmed=True,
                identity=person,session=session,kind=kind,split=row['split'],contract_sha256=contract_hash()))
            decide(root,row,'accepted');records.append(row)
    manifest=prepare(cfg,base)
    return cfg,base,root,manifest,records
