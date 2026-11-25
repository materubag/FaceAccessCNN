from types import SimpleNamespace
from unittest.mock import Mock
import json
import cv2
import numpy as np
from face_access.detection import Detection
from face_access.demo import run_demo


def test_demo_mirror_is_visual_and_invalid_roi_cannot_reuse_previous_crop(tmp_path,monkeypatch):
    frame=np.random.default_rng(7).integers(0,256,(240,320,3),dtype=np.uint8);before=frame.copy();crop=frame[50:146,60:156].copy()
    camera=Mock();camera.isOpened.return_value=True;camera.read.side_effect=[(True,frame),(True,frame)]
    model=SimpleNamespace(labels=['alice','bob','Desconocido'],checkpoint=dict(smoke_only=False,experimental=True,model_id='trained'),
        calibration=dict(threshold=.9,margin=.15),predict=Mock(return_value=np.array([.997,.001,.002])))
    pipeline=Mock();pipeline.process.side_effect=[
        dict(face_count=1,status='candidate',results=[dict(detection=Detection((60,50,96,96),.99),crop=crop,reasons=[])]),
        dict(face_count=1,status='rejected',results=[dict(detection=Detection((float('nan'),0,30,30),float('nan')),crop=None,reasons=['invalid_bbox'])])]
    monkeypatch.setattr('face_access.demo.Predictor',lambda *a:model);monkeypatch.setattr('face_access.demo.ProfilePipeline',lambda *a,**k:pipeline)
    monkeypatch.setattr('face_access.demo.cv2.VideoCapture',lambda _:camera);monkeypatch.setattr('face_access.demo.cv2.imshow',lambda *a:None)
    keys=iter([ord('m'),ord('q')]);monkeypatch.setattr('face_access.demo.cv2.waitKey',lambda _:next(keys))
    monkeypatch.setattr('face_access.demo.cv2.getWindowProperty',lambda *a:1);monkeypatch.setattr('face_access.demo.cv2.destroyAllWindows',lambda:None)
    events=tmp_path/'events.jsonl';run_demo('unused','unused',dict(camera=0,device='cpu',mirror=True,debug=False,consistent_frames=5,open_seconds=2.,cooldown_seconds=5.),events=events)
    assert model.predict.call_count==1
    np.testing.assert_array_equal(model.predict.call_args.args[0],crop)
    for call in pipeline.process.call_args_list:np.testing.assert_array_equal(call.args[0],before)
    np.testing.assert_array_equal(frame,before);camera.release.assert_called_once()
    rows=[json.loads(line) for line in events.read_text(encoding='utf-8').splitlines()]
    assert rows[-1]['state']=='QUALITY_BLOCKED' and rows[-1]['predicted'] is None


def test_demo_can_open_camera_with_directshow(monkeypatch):
    camera=Mock();camera.isOpened.return_value=True;camera.read.return_value=(False,None)
    capture=Mock(return_value=camera)
    monkeypatch.setattr('face_access.demo.cv2.VideoCapture',capture)
    monkeypatch.setattr('face_access.demo.ProfilePipeline',lambda *a,**k:Mock())
    monkeypatch.setattr('face_access.demo.cv2.destroyAllWindows',lambda:None)
    settings=dict(camera=1,camera_backend='dshow',device='cpu',mirror=True,debug=False,
        consistent_frames=5,open_seconds=2.,cooldown_seconds=5.)

    run_demo('unused','unused',settings,preview_only=True)

    capture.assert_called_once_with(1,cv2.CAP_DSHOW)


def test_demo_stream_without_video_keeps_waiting_and_keeps_barrier_closed(monkeypatch):
    from face_access.barrier import BarrierStatus
    frame=np.zeros((240,320,3),np.uint8)
    camera=Mock();camera.isOpened.return_value=True;camera.error='timed out';camera.read.side_effect=[(False,None),(True,frame)]
    barrier=Mock();barrier.status.return_value=BarrierStatus(ip='192.168.1.50')
    opened=[]
    monkeypatch.setattr('face_access.demo.MjpegCamera',lambda url:opened.append(url) or camera)
    monkeypatch.setattr('face_access.demo.BarrierLink',lambda host,port:barrier)
    model=SimpleNamespace(labels=['alice','Desconocido'],checkpoint=dict(smoke_only=False,experimental=False,model_id='trained'),
        calibration=dict(threshold=.9,margin=.15),predict=Mock())
    pipeline=Mock();pipeline.process.return_value=dict(face_count=0,status='no_face',results=[])
    monkeypatch.setattr('face_access.demo.Predictor',lambda *a:model);monkeypatch.setattr('face_access.demo.ProfilePipeline',lambda *a,**k:pipeline)
    shown=[];monkeypatch.setattr('face_access.demo.cv2.imshow',lambda name,view:shown.append(view.shape))
    keys=iter([0,ord('q')]);monkeypatch.setattr('face_access.demo.cv2.waitKey',lambda _:next(keys))
    monkeypatch.setattr('face_access.demo.cv2.getWindowProperty',lambda *a:1);monkeypatch.setattr('face_access.demo.cv2.destroyAllWindows',lambda:None)
    settings=dict(camera='http://camara.local:81/stream',barrier_host='pluma.local',barrier_port=4211,device='cpu',mirror=True,debug=False,
        consistent_frames=5,open_seconds=2.,cooldown_seconds=5.)

    run_demo('unused','unused',settings)

    assert opened==['http://camara.local:81/stream']
    # Primero la pantalla de "sin video" con el tamaño por defecto, luego el frame real.
    assert shown[0]==(480,640,3) and shown[1][:2]==(240,320)
    assert [c.args for c in barrier.set_open.call_args_list]==[(False,),(False,)]
    barrier.close.assert_called_once();camera.release.assert_called_once();model.predict.assert_not_called()
