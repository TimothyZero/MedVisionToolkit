#  Copyright (c) 2020. The Medical Image Computing (MIC) Lab, 陶豪毅
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import time
import os.path as osp
import torch
import os
import argparse
import numpy as np
from torch.cuda.amp import autocast
import SimpleITK as sitk
from tqdm import tqdm
import shutil

from medtk.utils import Config, get_root_logger
from medtk.data import build_dataloader, Viewer, ImageIO
from medtk.data.pipelines import *
from medtk.runner import build_optimizer, Runner


def merge_seg_pred(pred_results: list, method='mean'):
    pred_results = np.stack(pred_results, axis=0)
    if method == 'mean':
        pred_results = np.mean(pred_results, axis=0) > 0.5
    elif method == 'max':
        pred_results = np.max(pred_results, axis=0)
    return pred_results.astype(np.float32)


def merge_det_pred(pred_results: list, method='nms'):
    pass


def det_infer_net_loader(net, dataloader, infer_patch_size=8, as_rpn=False):
    net.eval()

    t = tqdm(dataloader)
    for i, data in enumerate(t):
        tic = time.time()

        batch_size = len(data['img_meta'])
        for j in range(batch_size):
            results = {'img_meta': data['img_meta'][j]}
            for k, v in data.items():
                if k != 'img_meta':
                    results[k] = v[j]

            img_key = 'img' if 'img' in data.keys() else 'patches_img'
            filename = results['img_meta']['filename']

            logger_dict = {}
            logger_dict.update({'filename': filename})
            t.set_postfix(logger_dict)

            dim = results['img_meta']['img_dim']
            if dim == 3:
                ext = 'nii.gz'
            else:
                ext = 'png'

            if 'patches_img' in results.keys():
                t.set_description(f'inferring on {len(results[img_key])} patches ...')
                for start in range(0, len(results[img_key]), infer_patch_size):
                    with autocast():
                        tensor_data = {'img': results[img_key][start: start + infer_patch_size].cuda(),
                                       'img_meta': None}
                        for k, v in results.items():
                            if k not in ['patches_img', 'img_meta']:
                                tensor_data[k.replace('patches_', '')] = v[start: start + infer_patch_size].cuda()

                        with torch.no_grad():
                            if hasattr(net, 'rpn_head') and as_rpn:
                                net_result = net.forward_infer(tensor_data, rpn=True)
                            else:
                                net_result = net.forward_infer(tensor_data)

                            if len(net_result) == 3:
                                prediction, anchor_id, _ = net_result
                            elif len(net_result) == 2:
                                prediction, _ = net_result
                                anchor_id = None

                    pred_bboxes = prediction.float()  # to float32, batch size != 1
                    for inner_p, patch_pred_bboxes in enumerate(pred_bboxes):
                        valid_patch_pred_bboxes = patch_pred_bboxes[patch_pred_bboxes[:, -1] != -1]
                        if anchor_id is not None:
                            patch_anchor_id = anchor_id[inner_p]
                            valid_patch_anchor_id = patch_anchor_id[patch_anchor_id != -1]
                            # print(valid_patch_anchor_id)
                            valid_patch_pred_bboxes = torch.cat([valid_patch_pred_bboxes, valid_patch_anchor_id[:,  None]], dim=1)
                        # valid_center = (valid_patch_pred_bboxes[:, :dim] + valid_patch_pred_bboxes[:, dim:2*dim]) / 2
                        # patch_center = torch.tensor(tensor_data['img'][0, 0].shape, device=valid_center.device) / 2
                        # valid_weight = 1 - torch.pow(valid_center - patch_center, 2).sum(dim=1).sqrt() / torch.pow(patch_center, 2).sum().sqrt()
                        # valid_patch_pred_bboxes[:, -1] = valid_patch_pred_bboxes[:, -1] * valid_weight.pow(0.25)
                        results['img_meta']['patches_pred_det'][start + inner_p] = valid_patch_pred_bboxes.cpu().numpy()
                    torch.cuda.empty_cache()

                logger_dict.update({'forward': time.time() - tic})
                t.set_postfix(logger_dict)
                results = reversed_pipeline(results)[0]
                results = saver(results)
                results['gt_det'] = results['pred_det']
                # print(len(results['pred_det']))
                # print(results['pred_det'])
                logger_dict.update({'total': time.time() - tic})
                t.set_postfix(logger_dict)
            # else:
            #     tensor_data = {'img':      torch.from_numpy(results['img']).unsqueeze(0).cuda(),
            #                    'img_meta': None}
            #     with torch.no_grad():
            #         prediction, _ = model.forward_infer(tensor_data)
            #     pred_bboxes = prediction.cpu().numpy()[0]
            #     pred_bboxes = pred_bboxes[pred_bboxes[:, -1] != -1]
            #     results['pred_det'] = pred_bboxes
            #
            #     results = monitor.result_pipeline(results)
            #     results = saver(results)
            #     results['gt_det'] = results['pred_det']
            #     print(len(results['pred_det']))
            #     print(results['pred_det'])
            # v(results)
        #         pred_results.append(results['pred_seg'])
        #         ImageIO.saveArray(osp.join(monitor.result_dir, f"{filename}_{multi_idx}_pred_seg.{ext}"),
        #                           results['pred_seg'], spacing=results['img_spacing'], origin=results['img_origin'])
        #
        # pred_results = merge_seg_pred(pred_results, method='mean')
        # ImageIO.saveArray(osp.join(monitor.result_dir, f"{filename}_pred_seg.{ext}"),
        #                   pred_results, spacing=results['img_spacing'], origin=results['img_origin'])
        # ImageIO.saveArray(osp.join(monitor.result_dir, f"{filename}_img.{ext}"),
        #                   results['img'], spacing=results['img_spacing'], origin=results['img_origin'])

        # if i >= 0:
        #     return


def build_task(cfg):
    assert cfg.TASK in ('SEG', 'CLS', 'DET')
    if cfg.TASK.upper() == 'SEG':
        model = cfg.model.to(device)
    elif cfg.TASK.upper() == 'CLS':
        model = cfg.model.to(device)
    elif cfg.TASK.upper() == 'DET':
        model = cfg.model.to(device)
    else:
        raise NotImplementedError
    return model


def build_loader(cfg, args):
    cfg.data[args.dataset].infer_mode = True
    dataset = cfg.data[args.dataset]
    if args.fold >= 0:
        if args.exclude:
            dataset.set_exclude_fold(args.fold)
        else:
            dataset.set_include_fold(args.fold)
    print(f'infer : {len(dataset)}')
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,  # must be 1
        workers_per_gpu=0,
        shuffle=False,
        drop_last=False
    )
    print(f'len dataset: {len(data_loader)} \n')
    return data_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='')
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--exclude', help='fold as exclude', action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='valid_infer')
    parser.add_argument('--rpn', help='infer using rpn', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # args.config = 'projects/LUNA2016/configs/cfg_v1_one_v1.py'
    # args.fold = 0
    # args.epoch = 150
    # args.dataset = 'infer'
    # # args.rpn = True
    # os.chdir('../../MedVisionProjects')

    cfg = Config.fromfile(args.config)
    cfg.gpus = 1
    if args.fold < 0:
        cfg.work_dir = osp.join(cfg.work_dir, cfg.module_name)
    else:
        cfg.work_dir = osp.join(cfg.work_dir, cfg.module_name, str(args.fold))
    if args.batch > 0:
        cfg.data.imgs_per_gpu = args.batch
    checkpoint = cfg.work_dir + f'/epoch_{args.epoch}.pth'
    if args.rpn:
        infer_results_dir = osp.join(cfg.work_dir, f'{args.dataset}_results_{args.epoch}ep_rpn')
    else:
        infer_results_dir = osp.join(cfg.work_dir, f'{args.dataset}_results_{args.epoch}ep')
    os.makedirs(infer_results_dir, exist_ok=True)

    shutil.copy(cfg.filename, osp.join(infer_results_dir, osp.basename(cfg.filename)))

    print(cfg.filename)
    print(checkpoint)
    print(infer_results_dir)

    """==============================================="""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_task(cfg)
    # monitor.setInferMode()
    # model.setHook()

    loader = build_loader(cfg, args)
    # dataset = build_infer_dataset(cfg, args)

    # put model on gpus
    # model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    model.load_state_dict(
        torch.load(checkpoint, map_location=device)["state_dict"])

    if cfg.TASK == 'DET':
        reversed_pipeline = BackwardCompose(cfg.infer_pipeline[::-1])
        saver = ForwardCompose([
            SaveFolder(infer_results_dir),
            # SaveImageToFile(ext='.nii.gz'),
            SaveAnnotations(with_det=True),
        ])
        print(saver)
        det_infer_net_loader(model, loader, 
                             infer_patch_size=cfg.data.imgs_per_gpu,
                             as_rpn=args.rpn)
    # elif cfg.TASK == 'SEG':
    #     saver = ForwardCompose([
    #         SaveFolder(monitor.result_dir),
    #         # SaveImageToFile(ext='same'),
    #         # SaveImageToFile(ext='.nii.gz'),
    #         SaveAnnotations(with_seg=True),
    #     ])
    #     print(saver)
    #     seg_infer_net(model, dataset)

    # infer_net_loader(model, loader, batch_processor)
