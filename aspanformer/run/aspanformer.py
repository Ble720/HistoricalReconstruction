from argparse import Namespace
import torch
import numpy as np
import cv2
import torch.nn.functional as F

import sys
sys.path.append('..')
from src.ASpanFormer.aspanformer import ASpanFormer as ASpanFormer_
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config

from base import Matching
from data_io import load_gray_scale_tensor_cv, load_gray_scale_tensor_cv_augment, load_pair_tensor

from data_fragments import load_fragments
from window import load_windows, load_fine_windows


class ASpanFormer(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)

        self.imsize = args.imsize
        self.match_threshold = args.match_threshold
        self.no_match_upscale = args.no_match_upscale
        self.online_resize = args.online_resize
        self.im_padding = False
        self.coarse_scale = args.coarse_scale
        self.eval_coarse = args.eval_coarse

        # Load model
        config = get_cfg_defaults()
        conf = lower_config(config)['aspan']
        conf['coarse']['train_res'] = args.train_res
        conf['coarse']['test_res'] = args.test_res
        conf['coarse']['coarsest_level'] = args.coarsest_level
        conf['match_coarse']['border_rm'] = args.border_rm
        conf['match_coarse']['thr'] = args.match_threshold

        if args.test_res:
            self.imsize = args.test_res[0]
            self.im_padding = args.test_res[0] == args.test_res[1]
        self.model = ASpanFormer_(config=conf)
        ckpt_dict = torch.load(args.ckpt)
        self.model.load_state_dict(ckpt_dict['state_dict'], strict=False)
        self.model = self.model.eval().to(self.device)

        # Name the method
        self.ckpt_name = args.ckpt.split('/')[-1].split('.')[0]
        self.name = f'ASpanFormer_{self.ckpt_name}'        
        if self.no_match_upscale:
            self.name += '_noms'
        print(f'Initialize {self.name} {args} ')

    def load_im(self, im_path, aug=False):
        if aug:
            return load_gray_scale_tensor_cv_augment(
                im_path,
                self.device,
                dfactor=8,
                imsize=self.imsize,
                value_to_scale=max,
                pad2sqr=self.im_padding
            )
        
        return load_gray_scale_tensor_cv(
            im_path,
            self.device,
            dfactor=8,
            imsize=self.imsize,
            value_to_scale=max,
            pad2sqr=self.im_padding
        )

    def match_inputs_(self, gray1, gray2, mask1=None, mask2=None):
        batch = {
            'image0': gray1, 'image1': gray2
        }
        if mask1 is not None and mask2 is not None and self.coarse_scale:
            ts_mask = F.interpolate(
                torch.cat([mask1, mask2])[None].float(),
                scale_factor=self.coarse_scale,
                mode='nearest',
                recompute_scale_factor=False
            )[0].bool().to(self.device)
            N = ts_mask.shape[0]
            split = int(N/2)
            ts_mask_1 = ts_mask[:split]
            ts_mask_2 = ts_mask[split:]
            batch.update({'mask0': ts_mask_1, 'mask1': ts_mask_2})

        # Forward pass
        self.model(batch, online_resize=self.online_resize)
        #print('bids ', fv_0[b_ids == 0].shape) fv_0, fv_1, b_ids = 
        #return fv_0, fv_1, b_ids
        # Output parsing
        if self.eval_coarse:
            kpts1 = batch['mkpts0_c'].cpu().numpy()
            kpts2 = batch['mkpts1_c'].cpu().numpy()
        else:
            kpts1 = batch['mkpts0_f'].cpu().numpy()
            kpts2 = batch['mkpts1_f'].cpu().numpy()
        scores = batch['mconf'].cpu().numpy()

        num_matches = batch['b_ids'] #change here b_ids shows which matches are for which pair

        return num_matches, kpts1, kpts2, scores


    def match_batch(self, q_path, r_paths, num_frags=1):
        N = len(r_paths) 
        gray1 = gray2 = None
        mask2 = None
        gray_q, _, mask_q = self.load_im(q_path, True)

        h, w = gray_q.shape

        gray1, ref_batch = load_fragments(
            gray_q.copy(),
            r_paths,
            num_frags=num_frags, 
            imsize=self.imsize,
            device=self.device,
            pad2sqr=self.im_padding,
            aug_q=False,
            aug_r=False
        )
        
        if mask_q is None:
            mask1 = None
        else:
            mask1 = mask_q.expand(N*num_frags, h, w)

        #print(gray1.shape, gray2.shape, mask1.shape, mask2.shape)
        
        matches, kpts1, kpts2, scores = self.match_inputs_(
            gray1, ref_batch['img'], mask1, ref_batch['mask']
        )
        torch.cuda.empty_cache()
        #print('fv0 ', fv_0.shape)
        #return fv_0, fv_1, b_ids
        #print(matches.shape)

        return matches, scores, ref_batch['id']

    def bin_ctcf(self, match, conf, bsize):
        counts = torch.bincount(match, minlength=bsize).cpu()
        confs = torch.bincount(match, weights=torch.tensor(conf).cuda(), minlength=bsize).cpu()
        return counts, confs

    def max_scores(self, counts, confs, ids):
        loop_ids = list(set(ids))
        size = len(loop_ids)
        max_counts = torch.zeros(size)
        max_confs = torch.zeros(size)

        for i, fid in enumerate(loop_ids):
            indexes = torch.where(torch.tensor(ids) == fid)
            max_counts[i] = max(counts[indexes])
            max_confs[i] = max(confs[indexes])

        return max_counts, max_confs

    def match_windows(self, q_path, r_paths, indexes, batch_size, prev_overflow=None):
        gray1 = gray2 = None
        mask2 = None
        gray_q, ratio, mask_q = self.load_im(q_path, True)

        h, w = gray_q.shape

        gray1, ref_batch, overflow, batch_info = load_windows(
            im_q=gray_q.copy(),
            q_ratio=ratio,
            r_paths=r_paths,
            r_indexes=indexes,
            device=self.device,
            prev_overflow=prev_overflow,
            batch_size=batch_size, 
            imsize=self.imsize
        )
        
        if mask_q is None:
            mask1 = None
        else:
            mask1 = mask_q.expand(gray1.shape[0], h, w)

        #print(gray1.shape, ref_batch['img'].shape, mask1.shape, ref_batch['mask'].shape)
        #print('overflow', overflow)
        
        matches, kpts1, kpts2, conf = self.match_inputs_(
            gray1, ref_batch['img'], mask1, ref_batch['mask']
        )
        torch.cuda.empty_cache()
        #print('fv0 ', fv_0.shape)
        #return fv_0, fv_1, b_ids
        #print(matches.shape)
        batch_info['size'] = ref_batch['img'].shape[0]

        return matches, conf, batch_info, ref_batch['id'], ref_batch['loc'], overflow



    def match_fine(self, q_path, r_paths, indexes, batch_size, curr_loc=None):
        gray1 = gray2 = None
        mask2 = None
        gray_q, ratio, mask_q = self.load_im(q_path, True)

        h, w = gray_q.shape

        overflow = None
        incomplete= {}
        incomplete['counts'] = incomplete['confs'] = None
        incomplete['id'] = []


        result = {}
        result['counts'] = torch.zeros(len(r_paths))
        result['confs'] = torch.zeros(len(r_paths))


        i = 0
        while i + indexes[0] < indexes[-1] + 1:
            #print('fine i', i)
            gray1, ref_batch, overflow = load_fine_windows(
                im_q=gray_q.copy(),
                q_ratio=ratio,
                r_paths=r_paths,
                r_indexes=indexes,
                curr_index=i,
                curr_loc=curr_loc,
                device=self.device,
                prev_overflow=overflow,
                batch_size=batch_size, 
                imsize=self.imsize
            )
        
            if mask_q is None:
                mask1 = None
            else:
                mask1 = mask_q.expand(gray1.shape[0], h, w)

            #print(gray1.shape, gray2.shape, mask1.shape, mask2.shape)
        
            matches, kpts1, kpts2, conf = self.match_inputs_(
                gray1, ref_batch['img'], mask1, ref_batch['mask']
            )

            counts, confs = self.bin_ctcf(matches, conf, gray1.shape[0])

            if overflow['index'] == -1:
                split = None
            else:
                split = ref_batch['id'].index(overflow['index'])
            
            if split != 0:
                curr_counts = counts[:split] if incomplete['counts'] is None else torch.cat((incomplete['counts'], counts[:split]))
                curr_confs = confs[:split] if incomplete['confs'] is None else torch.cat((incomplete['confs'], confs[:split]))
                curr_ids = incomplete['id'] + ref_batch['id'][:split]

                incomplete['counts'] = counts[split:]
                incomplete['confs'] = confs[split:]
                incomplete['id'] = ref_batch['id'][split:]

                max_counts, max_confs = self.max_scores(curr_counts, curr_confs, curr_ids)
                add_index = len(max_counts)
                result['counts'][i:i+add_index] = max_counts
                result['confs'][i:i+add_index] = max_confs
                i += add_index
            else:
                incomplete['counts'] = counts[split:] if incomplete['counts'] is None else torch.cat((incomplete['counts'], counts[split:]))
                incomplete['confs'] = confs[split:] if incomplete['confs'] is None else torch.cat((incomplete['confs'], confs[split:]))
                incomplete['id'] = ref_batch['id'][split:] if incomplete['id'] is None else incomplete['id'] + ref_batch['id'][split:]


        torch.cuda.empty_cache()
        #print('fv0 ', fv_0.shape)
        #return fv_0, fv_1, b_ids
        #print(matches.shape)

        return result


    def match_pairs(self, im1_path, im2_path):
        gray1, sc1, mask1 = self.load_im(im1_path)
        gray2, sc2, mask2 = self.load_im(im2_path)
        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(
            gray1, gray2, mask1, mask2
        )

        if self.no_match_upscale:
            return matches, kpts1, kpts2, scores, upscale.squeeze(0)

        # Upscale matches &  kpts
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
