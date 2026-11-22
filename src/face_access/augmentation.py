import cv2
import numpy as np
import torch

def augment_moderate(tensor,generator,policy):
    def uniform(a,b):return a+(b-a)*torch.rand((),generator=generator).item()
    image=tensor[0].numpy().copy();height,width=image.shape
    if uniform(0,1)<policy['horizontal_flip_probability']:image=image[:,::-1].copy()
    if uniform(0,1)<policy['affine_probability']:
        matrix=cv2.getRotationMatrix2D(((width-1)/2,(height-1)/2),uniform(-policy['rotation_degrees'],policy['rotation_degrees']),uniform(*policy['scale_range']))
        matrix[:,2]+=[uniform(-policy['translation_fraction'],policy['translation_fraction'])*width,uniform(-policy['translation_fraction'],policy['translation_fraction'])*height]
        image=cv2.warpAffine(image,matrix,(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    if uniform(0,1)<policy['brightness_probability']:image*=uniform(*policy['brightness_range'])
    if uniform(0,1)<policy['contrast_probability']:
        mean=image.mean();image=(image-mean)*uniform(*policy['contrast_range'])+mean
    return torch.from_numpy(np.ascontiguousarray(np.clip(image,0,1),dtype=np.float32)).unsqueeze(0)
