import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, './')
from stereo.datasets.dataset_template import DatasetTemplate
from stereo.modeling import build_trainer
from stereo.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='EFSNet KITTI test export')
    parser.add_argument('--cfg_file', type=str, default='cfgs/efsnet/efsnet_kitti.yaml')
    parser.add_argument('--data_cfg_file', type=str, default='cfgs/efsnet/kitti15_eval.yaml')
    parser.add_argument('--pretrained_model', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--pin_memory', action='store_true', default=False)
    args = parser.parse_args()

    cfgs = EasyDict(common_utils.config_loader(args.cfg_file))
    cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model
    data_cfgs = EasyDict(common_utils.config_loader(args.data_cfg_file))
    cfgs.DATA_CONFIG = data_cfgs.DATA_CONFIG
    cfgs.EVALUATOR = data_cfgs.EVALUATOR

    args.output_dir = args.output_dir or str(Path(args.pretrained_model).parent.parent)
    args.kitti_result_dir = os.path.join(args.output_dir, 'disp_0')
    args.dist_mode = False
    args.run_mode = 'eval'
    os.makedirs(args.kitti_result_dir, exist_ok=True)
    return args, cfgs


class KittiTestDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode='testing'):
        super().__init__(data_info, data_cfg, mode)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path = full_paths[:2]
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        sample = {
            'left': left_img,
            'right': right_img,
            'name': os.path.basename(left_img_path),
        }
        return self.transform(sample)


@torch.no_grad()
def main():
    args, cfgs = parse_config()
    local_rank = 0
    global_rank = 0
    torch.cuda.set_device(local_rank)

    log_file = os.path.join(args.output_dir, 'testkitti_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=local_rank)
    common_utils.log_configs(cfgs, logger=logger)

    data_info = cfgs.DATA_CONFIG.DATA_INFOS[0]
    test_dataset = KittiTestDataset(data_info, cfgs.DATA_CONFIG)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=args.pin_memory)
    logger.info('Total samples for KITTI test dataset: %d', len(test_dataset))

    model = build_trainer(args, cfgs, local_rank, global_rank, logger, None).model.cuda()
    model.eval()

    for i, data in enumerate(test_loader):
        for k, v in data.items():
            data[k] = v.to(local_rank) if torch.is_tensor(v) else v

        with torch.cuda.amp.autocast(enabled=cfgs.OPTIMIZATION.AMP):
            model_pred = model(data)

        disp_pred = model_pred['disp_pred'].squeeze(1)
        pad_top, pad_right, _, _ = [int(x.item()) if torch.is_tensor(x) else int(x) for x in data['pad']]
        disp_pred = disp_pred[:, pad_top:, :]
        if pad_right > 0:
            disp_pred = disp_pred[:, :, :-pad_right]

        img = (disp_pred.squeeze(0).cpu().numpy() * 256).astype('uint16')
        Image.fromarray(img).save(os.path.join(args.kitti_result_dir, data['name'][0]))
        logger.info('Iter:%4d/%d', i, len(test_loader))

    logger.info(args.kitti_result_dir)


if __name__ == '__main__':
    main()
