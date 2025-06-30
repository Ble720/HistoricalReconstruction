import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
import math

from augment import *

def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask

def fragment(image, size, num_frags):
    img_frags = []
    h, w = image.shape

    split = int(num_frags ** 0.5)

    height = h // split
    width = w // split
    for r in range(split):
        for c in range(split):
            img_frags.append(image[r*height:(r+1)*height, c*width:(c+1)*width])

    img_frags = np.stack(img_frags).transpose((1, 2, 0))

    scale = size/max(width, height)
    resize = (int(round(width * scale)), int(round(height*scale)))
    img_frags = cv2.resize(img_frags, resize)

    if len(img_frags.shape) == 2:
        img_frags = img_frags[None, :, :]
    else:
        img_frags = img_frags.transpose((2, 0, 1))
    return img_frags

def load_fragments(im_q, r_paths, num_frags, imsize, device, pad2sqr, aug_q=False, aug_r=False):
    ref_batch  = {}
    ref_batch['img'] = None
    ref_batch['mask'] = None
    ref_batch['id'] = []

    for i, rpath in enumerate(r_paths):
        im_r = cv2.imread(rpath, cv2.IMREAD_GRAYSCALE)
        im_r = fragment(im_r, imsize, num_frags)

        if aug_r:
            im_r = blur(im_r)

        _, ht_r, wt_r = im_r.shape
        mask_r = None
        if pad2sqr and (wt_r != ht_r):
            # Padding to square image
            im_r, mask_r = pad_bottom_right(im_r, imsize, ret_mask=True)
            
        elif mask_r == None:
            mask_r = np.ones((im_r.shape[0], imsize, imsize))

        ref_batch['img'] = im_r if ref_batch['img'] is None else np.concatenate((ref_batch['img'], im_r))
        ref_batch['mask'] = mask_r if ref_batch['mask'] is None else np.concatenate((ref_batch['mask'], mask_r))
        ref_batch['id'] += [i] * im_r.shape[0]

    if aug_q:
        pass
    
    im_q = transforms.functional.to_tensor(im_q).unsqueeze(0)
    q_batch = im_q.expand((ref_batch['img'].shape[0], 1, imsize, imsize)).to(device) 
    

    ref_batch['img'] = ref_batch['img'].transpose(((1, 2, 0)))
    ref_batch['img'] = transforms.functional.to_tensor(ref_batch['img']).unsqueeze(1)
    ref_batch['img'] = ref_batch['img'].float().to(device) #change this

    ref_batch['mask'] = torch.from_numpy(ref_batch['mask']).to(device)
    torch.cuda.empty_cache()

    return q_batch, ref_batch
