import argparse
import os
import sys

import numpy as np
import torch
from easydict import EasyDict
from PIL import Image

sys.path.insert(0, './')
from stereo.datasets.dataset_template import build_transform_by_cfg
from stereo.modeling import build_trainer
from stereo.utils import common_utils
from stereo.utils.disp_color import disp_to_color


def parse_config():
    parser = argparse.ArgumentParser(description='EFSNet inference')
    parser.add_argument('--cfg_file', type=str, default='cfgs/efsnet/efsnet_sceneflow.yaml')
    parser.add_argument('--left_img_path', type=str, required=True)
    parser.add_argument('--right_img_path', type=str, required=True)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--output', type=str, default='output/disp.png')
    args = parser.parse_args()

    cfgs = EasyDict(common_utils.config_loader(args.cfg_file))
    if args.pretrained_model:
        cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model
    args.dist_mode = False
    args.run_mode = 'infer'
    return args, cfgs


@torch.no_grad()
def main():
    args, cfgs = parse_config()
    local_rank = 0
    global_rank = 0
    torch.cuda.set_device(local_rank)

    logger = common_utils.create_logger(log_file=None, rank=local_rank)
    common_utils.log_configs(cfgs, logger=logger)

    trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, None)
    model = trainer.model
    model.eval()

    transform = build_transform_by_cfg(cfgs.DATA_CONFIG.DATA_TRANSFORM.EVALUATING)
    left_img = np.array(Image.open(args.left_img_path).convert('RGB'), dtype=np.float32)
    right_img = np.array(Image.open(args.right_img_path).convert('RGB'), dtype=np.float32)
    sample = transform({'left': left_img, 'right': right_img})
    sample['left'] = sample['left'].unsqueeze(0).cuda(local_rank)
    sample['right'] = sample['right'].unsqueeze(0).cuda(local_rank)

    with torch.cuda.amp.autocast(enabled=cfgs.OPTIMIZATION.AMP):
        model_pred = model(sample)

    disp = model_pred['disp_pred'].squeeze().detach().cpu().numpy()
    disp_color = disp_to_color(disp, max_disp=cfgs.MODEL.MAX_DISP).astype(np.uint8)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    Image.fromarray(disp_color).save(args.output)
    np.save(os.path.splitext(args.output)[0] + '.npy', disp)
    logger.info('Saved disparity visualization to %s', args.output)


if __name__ == '__main__':
    main()
