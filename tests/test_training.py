import copy
import numpy as np
import pytest
from face_access.common import read_csv,digest
from face_access.dataset import audit_manifest,validate_rows
from face_access.acquisition import decide
from face_access.training import train
from face_access.inference import Predictor
from face_access.evaluation import evaluate


def test_partitions_unknown_disjoint_and_reuse_blocked(dataset_case):
    cfg,base,root,manifest,_=dataset_case;rows,meta=audit_manifest(manifest)
    assert len(rows)==12 and meta['labels']==['alice','bob','Desconocido']
    bad=copy.deepcopy(rows);unknown=[r for r in bad if int(r['class_id'])==2]
    unknown[-1]['identity']=unknown[0]['identity']
    with pytest.raises(ValueError,match='Fuga'):validate_rows(bad,meta['labels'])
    bad=copy.deepcopy(rows);bad[-1]['sha256']=bad[0]['sha256']
    with pytest.raises(ValueError,match='Duplicado'):validate_rows(bad,meta['labels'])


def test_human_revocation_blocks_manifest(dataset_case):
    _,_,root,manifest,records=dataset_case;decide(root,records[0],'rejected')
    with pytest.raises(ValueError,match='revocada'):audit_manifest(manifest)


def test_capture_requires_consent_and_session_before_any_write(tmp_path):
    from face_access.acquisition import capture
    with pytest.raises(ValueError,match='consent'):capture('new_person','sesion_01',data_root=tmp_path/'absent')
    with pytest.raises(ValueError,match='independent'):capture('new_person','sesion_01',consent=True,data_root=tmp_path/'absent')
    assert not (tmp_path/'absent').exists()


def test_capture_video_sampling_provenance_and_pending_review(tmp_path,monkeypatch):
    from face_access import acquisition
    from face_access.detection import Detection
    source=tmp_path/'source.mp4';source.write_bytes(b'synthetic mock video: no real recording')
    frame=np.random.default_rng(1).integers(0,256,(96,96,3),dtype=np.uint8)
    class Camera:
        def __init__(self):self.index=0;self.released=False
        def isOpened(self):return True
        def get(self,key):return 60.
        def read(self):
            self.index+=1
            return (True,frame) if self.index<=61 else (False,None)
        def release(self):self.released=True
    camera=Camera()
    class Pipeline:
        def process(self,frame):return dict(face_count=1,status='candidate',results=[dict(crop=frame,detection=Detection((0,0,96,96),.99))])
    monkeypatch.setattr(acquisition.cv2,'VideoCapture',lambda _:camera)
    monkeypatch.setattr(acquisition,'ProfilePipeline',lambda *args,**kwargs:Pipeline())
    folder=acquisition.capture('new_person','sesion_01',str(source),tmp_path/'data',consent=True,independent=True,preview=False)
    from face_access.common import read_json
    rows=read_json(folder/'inventory.json');metadata=read_json(folder/'run.json')
    assert camera.released and 3<=metadata['sampled_frames']<=4
    assert rows[0]['source_frame']==0 and rows[-1]['source_frame']>=40
    assert not (tmp_path/'data/reviews.jsonl').exists()


def test_one_epoch_save_load_validate_calibrate_and_single_test_guard(dataset_case):
    cfg,base,_,manifest,_=dataset_case
    checkpoint=train(cfg,manifest,base/'run',smoke=False,device='cpu')
    model=Predictor(checkpoint,device='cpu');assert model.model.classifier[-1].out_features==3
    metrics=evaluate(checkpoint,manifest,base/'val',device='cpu');assert metrics['split']=='val'
    calibrated=Predictor(checkpoint,base/'val/calibration.json','cpu');assert calibrated.calibration['calibrated_split']=='val'
    with pytest.raises(ValueError,match='confirm-frozen'):evaluate(checkpoint,manifest,base/'test',split='test',calibration=base/'val/calibration.json',device='cpu')
    result=evaluate(checkpoint,manifest,base/'test',split='test',calibration=base/'val/calibration.json',confirm_frozen=True,device='cpu')
    assert result['samples']==4
    with pytest.raises(FileExistsError):evaluate(checkpoint,manifest,base/'test_again',split='test',calibration=base/'val/calibration.json',confirm_frozen=True,device='cpu')
    with pytest.raises(FileExistsError):train(cfg,manifest,base/'run',device='cpu')


def test_smoke_is_flagged_and_not_usable_for_access(dataset_case):
    from face_access.demo import run_demo
    cfg,base,_,manifest,_=dataset_case;checkpoint=train(cfg,manifest,base/'smoke',smoke=True,device='cpu')
    evaluate(checkpoint,manifest,base/'smoke_val',device='cpu')
    with pytest.raises(ValueError,match='smoke'):run_demo(checkpoint,base/'smoke_val/calibration.json',dict(device='cpu'))
