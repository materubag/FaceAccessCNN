import numpy as np
import pytest
import torch
from face_access.preprocessing import base_array
from face_access.inference import preprocess
from face_access.detection import Detection,ProfilePipeline,square_crop,load_profile
from face_access.model import FaceCNN


def test_shape_grayscale_normalization_and_invalid_crop():
    image=np.full((96,96,3),255,np.uint8);tensor=preprocess(image)
    assert tensor.shape==(1,96,96) and tensor.dtype==torch.float32
    assert tensor.min()==tensor.max()==1
    np.testing.assert_array_equal(tensor[0].numpy(),base_array(image,'median'))
    with pytest.raises(ValueError):preprocess(np.zeros((80,80,3),np.uint8))
    with pytest.raises(ValueError):preprocess(np.ones((96,96,3),np.float32))


def test_no_multiple_and_ambiguous_faces_are_not_candidates():
    frame=np.random.default_rng(2).integers(0,256,(240,320,3),dtype=np.uint8)
    class Fake:
        def __init__(self,rows):self.rows=rows
        def detect(self,frame):return self.rows
    good=Detection((50,40,100,140),.99,np.array([[80,85],[125,85],[105,110],[85,145],[125,145]],float))
    assert ProfilePipeline('yunet_v1',Fake([]),False).process(frame)['status']=='no_face'
    assert ProfilePipeline('yunet_v1',Fake([good]),False).process(frame)['status']=='candidate'
    assert ProfilePipeline('yunet_v1',Fake([good,good]),False).process(frame)['face_count']==2
    weak=Detection(good.box,.7,good.landmarks)
    assert ProfilePipeline('yunet_v1',Fake([weak]),False).process(frame)['status']=='review'
    bad=Detection((-100,-100,1,1),.99,good.landmarks)
    assert ProfilePipeline('yunet_v1',Fake([bad]),False).process(frame)['status']=='rejected'


def test_crop_truncation_and_dimensions():
    image=np.zeros((240,320,3),np.uint8);cfg=load_profile()
    crop,meta=square_crop(image,(50,40,100,140),cfg)
    assert crop.shape==(96,96,3) and meta['padding_fraction']==0
    with pytest.raises(ValueError):square_crop(image,(-400,-400,30,30),cfg)


def test_original_architecture_parameters_and_dynamic_new_class():
    original=FaceCNN(head='flatten');assert sum(p.numel() for p in original.parameters())==1272644
    gap=FaceCNN(head='gap');assert sum(p.numel() for p in gap.parameters())==101188
    with torch.inference_mode():assert original(torch.zeros(2,1,96,96)).shape==(2,4)
    new=FaceCNN(head='flatten',num_classes=5)
    with torch.inference_mode():assert new(torch.zeros(1,1,96,96)).shape==(1,5)


def test_unreadable_image_is_explicit(tmp_path):
    from face_access.preprocessing import read_image
    path=tmp_path/'empty.jpg';path.write_bytes(b'')
    with pytest.raises(ValueError,match='vacía'):read_image(path)
