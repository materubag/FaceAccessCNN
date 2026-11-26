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


def test_supervised_actuation_preserves_checks_and_closes_on_stream_loss(tmp_path,monkeypatch):
    from face_access.demo import run_demo
    from face_access.detection import Detection
    from face_access.barrier import BarrierStatus
    from copy import deepcopy
    frame=np.zeros((240,320,3),np.uint8);crop=np.zeros((96,96,3),np.uint8)
    candidate=dict(face_count=1,status='candidate',results=[dict(detection=Detection((60,50,96,96),.99),crop=crop,reasons=[])])
    empty=dict(face_count=0,status='no_face',results=[])
    multiple=dict(face_count=2,status='multiple_faces_review',results=[dict(detection=Detection((60,50,96,96),.99),crop=crop,reasons=['multiple'])]*2)
    review=deepcopy(candidate);review['status']='review';review['results'][0]['reasons']=['blurry']
    detections=[deepcopy(candidate) for _ in range(5)]+[empty,deepcopy(candidate),deepcopy(candidate),deepcopy(candidate),deepcopy(candidate),multiple,review]
    camera=Mock();camera.isOpened.return_value=True;camera.error='WiFi lost'
    camera.read.side_effect=[(True,frame)]*len(detections)+[(False,None)]
    config=dict(consistent_frames=5,recognition_seconds=1.5,open_seconds=2.,cooldown_seconds=5.)
    model=SimpleNamespace(authorization_enabled=False,labels=['Persona 1','Persona 2','Persona 3','Desconocido'],
        checkpoint=dict(smoke_only=False,experimental=True,model_id='experiment',temporal_policy=config),
        calibration=dict(threshold=.9172210097312927,margin=.15),
        predict=Mock(side_effect=[np.array([.999,.0003,.0003,.0004])]*6+
            [np.array([.0003,.999,.0003,.0004]),np.array([.50,.49,.005,.005]),np.array([.0003,.0003,.0004,.999])]))
    original=deepcopy(model.checkpoint);cal_before=deepcopy(model.calibration)
    pipeline=Mock();pipeline.process.side_effect=detections
    barrier=Mock();barrier.status.return_value=BarrierStatus(ip='192.168.1.50')
    factory=Mock(return_value=barrier)
    monkeypatch.setattr('face_access.demo.Predictor',lambda *a:model)
    monkeypatch.setattr('face_access.demo.ProfilePipeline',lambda *a,**k:pipeline)
    monkeypatch.setattr('face_access.demo.MjpegCamera',lambda _:camera)
    monkeypatch.setattr('face_access.demo.BarrierLink',factory)
    monkeypatch.setattr('face_access.demo.cv2.destroyAllWindows',lambda:None)
    times=iter(i*.4 for i in range(len(detections)+1))
    monkeypatch.setattr('face_access.demo.time.monotonic',lambda:next(times))
    display=iter([True]*len(detections)+[False]);monkeypatch.setattr('face_access.demo.show',lambda *a:next(display))
    settings=dict(camera='http://camara.local:81/stream',device='cpu',consistent_frames=5,recognition_seconds=0.,open_seconds=2.,
        cooldown_seconds=5.,mirror=True,debug=False,barrier_host='pluma.local',barrier_port=4211,supervised_actuation=True)
    events=tmp_path/'events.jsonl';run_demo('unused','unused',settings,events=events)
    factory.assert_called_once_with('pluma.local',4211)
    opened=[call.args[0] for call in barrier.set_open.call_args_list]
    assert opened[:5]==[False,False,False,False,True]
    assert not any(opened[5:])  # no face, identity change, ambiguous, unknown, multiple, quality, stream loss
    barrier.close.assert_called_once();camera.release.assert_called_once()
    assert model.authorization_enabled is False and model.checkpoint==original and model.calibration==cal_before
    logs=[json.loads(line) for line in events.read_text(encoding='utf-8').splitlines()]
    assert all(r['supervised_actuation'] and not r['model_approval_passed'] for r in logs)
    assert any(r['authorization'] for r in logs)


def test_supervised_flag_is_explicit_and_not_preview(monkeypatch):
    from face_access.cli import parser,main
    assert parser().parse_args(['demo']).supervised_actuation is False
    assert parser().parse_args(['demo','--supervised-actuation']).supervised_actuation is True
    with pytest.raises(SystemExit):main(['demo','--preview-only','--supervised-actuation'])


@pytest.mark.parametrize('missing_calibration,smoke',[(True,False),(False,True)])
def test_supervised_actuation_still_rejects_smoke_and_missing_calibration(monkeypatch,missing_calibration,smoke):
    from face_access.demo import run_demo
    model=SimpleNamespace(checkpoint={'smoke_only':smoke},calibration=None if missing_calibration else {'threshold':.9,'margin':.15})
    monkeypatch.setattr('face_access.demo.Predictor',lambda *a:model)
    with pytest.raises(ValueError):run_demo('unused','unused',dict(device='cpu',supervised_actuation=True))
