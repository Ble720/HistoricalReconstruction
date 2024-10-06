import os
from copy import deepcopy
from src.utils.dataset import read_megadepth_gray
import shutil

import torch
import numpy as np

from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

from utils import get_paths, check_matches

# You can also change the default values like thr. and npe (based on input image size)

def match():
    model_type = 'full' #['full', 'opt']
    precision = 'fp32' #['fp32', 'mp', 'fp16']
    batch_size = 64
    img_size = 256
    topk = 15
    save = './save_eloftr_raw'

    if model_type == 'full':
        _default_cfg = deepcopy(full_default_cfg)
    elif model_type == 'opt':
        _default_cfg = deepcopy(opt_default_cfg)
        
    if precision == 'mp':
        _default_cfg['mp'] = True
    elif precision == 'fp16':
        _default_cfg['half'] = True
        
    matcher = LoFTR(config=_default_cfg)

    matcher.load_state_dict(torch.load("weights/eloftr_outdoor.ckpt")['state_dict'])
    matcher = reparameter(matcher) # no reparameterization will lead to low performance

    if precision == 'fp16':
        matcher = matcher.half()

    matcher = matcher.eval().cuda()

    query_path = get_paths('../image-matching-toolbox/src_100')
    ref_path = get_paths('../image-matching-toolbox/target')

    ref_iter = len(ref_path)/batch_size
    if int(ref_iter) != ref_iter:
        ref_iter += 1
    ref_iter = int(ref_iter)

    for i, qpath in enumerate(query_path):
        
        q_img, q_mask, _ = read_megadepth_gray(qpath, resize=img_size, padding=True)
        query_batch = q_img.expand(batch_size, 1, q_img.shape[1], q_img.shape[2]).cuda()
        #q_mask_batch = q_mask.expand(batch_size, img_size, img_size).cuda()
        t = 0
        score = torch.zeros(len(ref_path))
        while t < ref_iter:
            #print(f'iter {t}/{ref_iter}')
            if t == ref_iter-1:
                ref_batch_paths = ref_path[t*batch_size:]
                query_batch = query_batch[:len(ref_batch_paths)]
                #q_mask_batch = q_mask_batch[:len(ref_batch_paths)]
            else:
                ref_batch_paths = ref_path[t*batch_size:(t+1)*batch_size]
            
            ref_batch = []
            ref_mask_batch = []

            for rpath in ref_batch_paths:
                r_img, r_mask, _ = read_megadepth_gray(rpath, resize=img_size, padding=True)
                ref_batch.append(r_img)
                #ref_mask_batch.append(r_mask)
            ref_batch = torch.stack(ref_batch, dim=0).cuda()
            ref_mask_batch = torch.stack(ref_mask_batch, dim=0).cuda()

            batch = {'image0': query_batch, 'image1': ref_batch}#, 'mask0': q_mask_batch, 'mask1': ref_mask_batch}

            with torch.no_grad():
                if precision == 'mp':
                    with torch.autocast(enabled=True, device_type='cuda'):
                        matcher(batch)
                else:
                    matcher(batch)
                #mkpts0 = batch['mkpts0_f'].cpu().numpy()
                #mconf = batch['mconf'].cpu().numpy()
                count = torch.bincount(batch['b_ids'])

            #counts = torch.bincount(mkpts0)
            score[t*batch_size:t*batch_size+len(count)] = count
            t += 1
        print(f'Calc top {topk}')
        _, max_arg = torch.topk(score, topk, largest=True)

        si_name = query_path[i].replace('\\', '/').split('/')[-1]

        dir_name = '{}/{}'.format(save, si_name[:-4])
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        shutil.copyfile(query_path[i], '{}/{}/SRC_{}'.format(save, si_name[:-4], si_name))

        for ip in max_arg:
            ti_name = ref_path[ip.item()].replace('\\', '/').split('/')[-1]
            shutil.copyfile(ref_path[ip.item()], '{}/{}/{}'.format(save, si_name[:-4], ti_name))
        print(f"source {i+1}/{len(query_path)} done\n")
            
# if precision == 'fp16':
#     img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
#     img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
# else:
#     img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
#     img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.


# Inference with EfficientLoFTR and get prediction

if __name__ == '__main__':
    match()