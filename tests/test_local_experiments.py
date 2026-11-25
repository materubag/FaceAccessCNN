import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
import torch
from face_access.common import canonical,digest,save_json
from face_access.config import detector_profile,PREPROCESSING
from face_access.inference import import_model,Predictor
from face_access.model import FaceCNN
from face_access.door import DoorState


@pytest.fixture
def legacy(tmp_path):
    model_cfg=dict(channels=[2,4,8],hidden=8,dropout=.3,head='flatten')
    model=FaceCNN(**model_cfg).eval()
    contract=dict(preprocessing=PREPROCESSING,detection_profile=detector_profile())
    path=tmp_path/'source.pt'
    torch.save(dict(mode='real',smoke_only=False,weights=model.state_dict(),
        labels=['Persona 1','Persona 2','Persona 3','Desconocido'],manifest_sha256='synthetic',
        config=dict(model=model_cfg,decision=dict(consistent_frames=5,open_seconds=2.,cooldown_seconds=5.),safety_gate_criteria={'minimum':30}),
        input_contract=contract,input_contract_sha256=canonical(contract)),path)
    cal=tmp_path/'source_cal.json'
    save_json(cal,dict(checkpoint_sha256=digest(path),manifest_sha256='synthetic',calibrated_split='val',threshold=.96,margin=.15))
    return path,cal,model


def test_conversion_preserves_weights_predictions_and_calibration(legacy,tmp_path):
    source,cal,model=legacy;path,targetcal=import_model(source,cal,tmp_path/'models','experiment')
    imported=Predictor(path,targetcal,'cpu')
    image=np.random.default_rng(42).integers(0,256,(96,96,3),dtype=np.uint8)
    from face_access.inference import preprocess
    with torch.inference_mode():expected=model(preprocess(image)[None]).softmax(1)[0].numpy()
    np.testing.assert_array_equal(expected,imported.predict(image))
    assert digest(cal)==digest(targetcal)
    assert imported.labels==['Persona 1','Persona 2','Persona 3','Desconocido']
    assert imported.authorization_enabled is False
    assert 'config' not in imported.checkpoint and 'manifest' not in imported.checkpoint
    with pytest.raises(FileExistsError):import_model(source,cal,tmp_path/'models','experiment')


def test_calibration_change_is_rejected(legacy,tmp_path):
    source,cal,_=legacy;path,targetcal=import_model(source,cal,tmp_path/'models','experiment')
    changed=json.loads(targetcal.read_text(encoding='utf-8'));changed['threshold']=.5;save_json(targetcal,changed)
    with pytest.raises(ValueError,match='congelada'):Predictor(path,targetcal,'cpu')


def test_failed_attestation_cannot_enable_door(legacy,tmp_path):
    source,cal,_=legacy;gate=tmp_path/'gate.json'
    save_json(gate,dict(allowed=False,checks={'acceptance':False}))
    path,targetcal=import_model(source,cal,tmp_path/'models','experiment',gate)
    assert Predictor(path,targetcal,'cpu').authorization_enabled is False
    door=DoorState(authorization_enabled=False)
    for i in range(100):
        s=door.step(0,1,i*.1)
        assert not s.opened and not s.authorization_event and s.valid_predictions==0


def test_diagnostic_demo_never_connects_to_physical_barrier(monkeypatch):
    from face_access.demo import run_demo
    model=SimpleNamespace(authorization_enabled=False,labels=['Persona 1','Persona 2','Persona 3','Desconocido'],
        checkpoint=dict(smoke_only=False,experimental=True,model_id='experiment'),calibration=dict(threshold=.96,margin=.15))
    camera=Mock();camera.isOpened.return_value=True;camera.read.return_value=(False,None)
    barrier=Mock()
    monkeypatch.setattr('face_access.demo.Predictor',lambda *a:model)
    monkeypatch.setattr('face_access.demo.ProfilePipeline',lambda *a,**k:Mock())
    monkeypatch.setattr('face_access.demo.cv2.VideoCapture',lambda *a:camera)
    monkeypatch.setattr('face_access.demo.cv2.destroyAllWindows',lambda:None)
    monkeypatch.setattr('face_access.demo.BarrierLink',barrier)
    run_demo('unused','unused',dict(camera=0,device='cpu',consistent_frames=5,open_seconds=2.,cooldown_seconds=5.,barrier_host='pluma.local'))
    barrier.assert_not_called();camera.release.assert_called_once()
