import json
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
import torch
from face_access.common import digest,save_json
from face_access.config import input_contract,contract_hash
from face_access.inference import Predictor,classification,decisions,weights_digest,FORMAT,MODELS
from face_access.model import FaceCNN
from face_access.display import render_demo,display_box,display_point
from face_access.door import DoorState
from face_access.cli import main


@pytest.fixture
def artifact(tmp_path):
    model_cfg=dict(channels=[2,4,8],hidden=8,head='flatten',num_classes=3)
    model=FaceCNN(**model_cfg);state=model.state_dict();path=tmp_path/'model.pt'
    torch.save(dict(format=FORMAT,weights=state,weights_sha256=weights_digest(state),model_config=model_cfg,labels=['alice','bob','Desconocido'],
        input_contract=input_contract(),contract_sha256=contract_hash(),manifest_sha256='synthetic',smoke_only=False,experimental=True,model_id='trained'),path)
    cal=tmp_path/'cal.json';save_json(cal,dict(checkpoint_sha256=digest(path),manifest_sha256='synthetic',calibrated_split='val',threshold=.9,margin=.15))
    return path,cal


def test_load_predict_calibration_and_invalid_binding(artifact):
    path,cal=artifact;model=Predictor(path,cal,'cpu');image=np.random.default_rng(1).integers(0,256,(96,96,3),dtype=np.uint8)
    p=model.predict(image);assert p.shape==(3,) and np.isclose(p.sum(),1)
    record=json.loads(cal.read_text(encoding='utf-8'));record.pop('checkpoint_sha256');save_json(cal,record)
    with pytest.raises(ValueError,match='calibración'):Predictor(path,cal,'cpu')


def test_model_selection_and_missing_private_weights(tmp_path,capsys):
    assert MODELS['r03b']['threshold']==.9813899397850037 and MODELS['r02']['threshold']==.9197969436645508
    main(['models','list']);assert 'r03b' in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):Predictor(tmp_path/'missing.pt')


def test_unknown_score_margin_strict_threshold_and_invalid_numbers():
    assert decisions(np.array([[.97,.02,.01]]),.9,.15)[0]==0
    assert decisions(np.array([[.01,.01,.98]]),.9,.15)[0]==2
    assert decisions(np.array([[.45,.44,.11]]),.4,.15)[0]==2
    assert decisions(np.array([[.9,.05,.05]]),.9,.15)[0]==2
    assert classification(None,.9,.15,3)['predicted'] is None
    assert classification(np.array([np.nan,0,1]),.9,.15,3)['label']==2
    with pytest.raises(ValueError):decisions(np.array([[2.,0,0]]),.9)


def test_visual_mirror_cannot_change_roi_tensor_predictions(artifact):
    model=Predictor(*artifact,device='cpu');frame=np.random.default_rng(8).integers(0,256,(240,320,3),dtype=np.uint8);before=frame.copy()
    crop=frame[50:146,60:156].copy();a=model.predict(crop)
    details=classification(a,.9,.15,3);quality=dict(status='candidate',ok=True,reasons=[],detector_score=.99,threshold=.9,minimum_margin=.15)
    for mirror in (False,True):render_demo(frame,[(60,50,96,96)],DoorState().step(0,1,0),details,quality,1,mirror=mirror,labels=model.labels)
    np.testing.assert_array_equal(a,model.predict(crop));np.testing.assert_array_equal(frame,before)
    assert display_box((10,20,30,40),100)==(60,20,30,40);assert display_point((10,20),100)==(89,20)


def test_smoke_check_and_camera_failure_release(artifact,monkeypatch):
    from face_access.demo import run_demo
    path,cal=artifact;camera=Mock();camera.isOpened.return_value=False
    monkeypatch.setattr('face_access.demo.cv2.VideoCapture',lambda _:camera);monkeypatch.setattr('face_access.demo.cv2.destroyAllWindows',lambda:None)
    settings=dict(camera=0,device='cpu',mirror=True,debug=False,consistent_frames=5,open_seconds=2.,cooldown_seconds=5.)
    with pytest.raises(RuntimeError,match='Webcam'):run_demo(path,cal,settings)
    camera.release.assert_called_once()
