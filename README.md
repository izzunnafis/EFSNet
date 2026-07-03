# EFSNet

Official implementation of **EFSNet: Lightweight stereo matching through hybrid cost volume fusion and multi-scale refinement**.

- **Authors:** Izzun Nafis Ibadik and Egi Muhammad Idris Hidayat
- **Journal:** International Journal of Cognitive Computing in Engineering, Vol. 7, pp. 631-643, 2026
- **DOI:** https://doi.org/10.1016/j.ijcce.2026.06.001

- **Links:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666307426000185), [KITTI 2012](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo&error=3&eval=est), [KITTI 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo&eval_area=est&eval_gt=noc)

EFSNet is a lightweight stereo matching network for dense disparity estimation from rectified stereo image pairs. The method targets real-time 3D perception by combining efficient feature extraction, hybrid 2D-3D cost aggregation, multi-scale refinement, and a LinearLog training loss.

## Highlights

- **Hybrid 2D-3D aggregation:** combines 3D cost-volume reasoning with efficient 2D aggregation.
- **Multi-scale refinement:** refines disparity using a dual-path refinement design for image detail and low-texture robustness.
- **LinearLog loss:** combines logarithmic and linear penalty behavior to train on both small and large disparity errors.
- **Two model sizes:** EFSNet is the standard model; EFSNet-lite reduces repeated aggregation blocks for the lightweight setting.

## Model Variants

The standard and lite variants share the same high-level architecture. The main exposed architectural difference is the number of repeated aggregation blocks:

| Variant | Configs | `AGGREGATION_BLOCKS` |
| --- | --- | --- |
| EFSNet | `efsnet_sceneflow.yaml`, `efsnet_kitti.yaml`, `efsnet_manual.yaml` | `[4, 4, 2]` |
| EFSNet-lite | `efsnet_lite_kitti.yaml`, `efsnet_lite_manual.yaml` | `[1, 1, 1]` |

`EXPANSE_RATIO` is separate from the repeat count. It controls the MobileNetV2-style inverted residual expansion ratio inside the aggregation blocks.

## Reported Results

SceneFlow results reported for the paper:

| Method | Params (M) | FLOPs (G) | EPE (px) | Runtime (ms) |
| --- | ---: | ---: | ---: | ---: |
| EFSNet-lite | 3.35 | 26.57 | 0.72 | 16 |
| EFSNet | 4.59 | 29.93 | 0.64 | 18 |

Official KITTI benchmark entries are available for EFSNet and EFSNet-lite on KITTI 2012 and KITTI 2015.

## Installation

Create an environment:

```bash
conda create -n efsnet python=3.8
conda activate efsnet
```

Install PyTorch for your CUDA version from https://pytorch.org/get-started/locally/, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Layout

Set dataset roots in the YAML config files or place datasets under the default paths:

```text
data/
  SceneFlow/
    sceneflow_finalpass_train.txt
    sceneflow_finalpass_test.txt
    ...
  KITTI12/
    kitti12_train194.txt
    kitti12_val14.txt
    kitti12_test.txt
    kitti12/
      training/
      testing/
  KITTI15/
    kitti15_train200.txt
    kitti15_val20.txt
    kitti15_test.txt
    kitti15/
      training/
      testing/
  manual/
    zed_file_train.txt
    zed_file_val.txt
```

Manual dataset split lines use:

```text
left_image_path right_image_path disparity_numpy_path
```

The disparity path is loaded with `numpy.load`.

## Training

Train the standard EFSNet model on SceneFlow:

```bash
python tools/train.py --cfg_file cfgs/efsnet/efsnet_sceneflow.yaml
```

Fine-tune the standard EFSNet model on KITTI:

```bash
python tools/train.py \
  --cfg_file cfgs/efsnet/efsnet_kitti.yaml \
  --pretrained_model path/to/sceneflow_checkpoint.pth \
  --extra_tag kitti_finetune
```

Train or fine-tune the lite variant by selecting a lite config:

```bash
python tools/train.py \
  --cfg_file cfgs/efsnet/efsnet_lite_kitti.yaml \
  --pretrained_model path/to/sceneflow_checkpoint.pth \
  --extra_tag kitti_lite_finetune
```

For distributed training:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:23456 \
  tools/train.py --dist_mode --cfg_file cfgs/efsnet/efsnet_sceneflow.yaml
```

## Evaluation

Evaluate on SceneFlow:

```bash
python tools/eval.py \
  --cfg_file cfgs/efsnet/efsnet_sceneflow.yaml \
  --pretrained_model path/to/checkpoint.pth
```

Evaluate with KITTI 2015 data settings:

```bash
python tools/eval.py \
  --cfg_file cfgs/efsnet/efsnet_kitti.yaml \
  --eval_data_cfg_file cfgs/efsnet/kitti15_eval.yaml \
  --pretrained_model path/to/checkpoint.pth
```

Evaluate EFSNet-lite by using the lite model config:

```bash
python tools/eval.py \
  --cfg_file cfgs/efsnet/efsnet_lite_kitti.yaml \
  --eval_data_cfg_file cfgs/efsnet/kitti15_eval.yaml \
  --pretrained_model path/to/lite_checkpoint.pth
```

## Inference

Run inference on one stereo pair:

```bash
python tools/infer.py \
  --cfg_file cfgs/efsnet/efsnet_sceneflow.yaml \
  --pretrained_model path/to/checkpoint.pth \
  --left_img_path path/to/left.png \
  --right_img_path path/to/right.png \
  --output output/disp.png
```

This writes both `output/disp.png` and `output/disp.npy`.

## KITTI Test Export

Generate KITTI-style `disp_0` predictions:

```bash
python tools/test_kitti.py \
  --cfg_file cfgs/efsnet/efsnet_kitti.yaml \
  --data_cfg_file cfgs/efsnet/kitti15_eval.yaml \
  --pretrained_model path/to/checkpoint.pth \
  --output_dir output/kitti_submission
```

## Citation

If this code or paper is useful for your work, please cite:

```bibtex
@article{ibadik2026efsnet,
  title={EFSNet: Lightweight stereo matching through hybrid cost volume fusion and multi-scale refinement},
  author={Ibadik, Izzun Nafis and Hidayat, Egi Muhammad Idris},
  journal={International Journal of Cognitive Computing in Engineering},
  volume={7},
  pages={631--643},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.ijcce.2026.06.001}
}
```

## Acknowledgements

This implementation builds on the OpenStereo stereo matching framework and the LightStereo line of efficient stereo models. Please also cite the relevant OpenStereo/LightStereo work when using this repository:

```bibtex
@article{guo2023openstereo,
  title={OpenStereo: A Comprehensive Benchmark for Stereo Matching and Strong Baseline},
  author={Guo, Xianda and Zhang, Chenming and Lu, Juntao and Wang, Yiqi and Duan, Yiqun and Yang, Tian and Zhu, Zheng and Chen, Long},
  journal={arXiv preprint arXiv:2312.00343},
  year={2023}
}

@article{guo2024lightstereo,
  title={LightStereo: Channel Boost Is All You Need for Efficient 2D Cost Aggregation},
  author={Guo, Xianda and Zhang, Chenming and Nie, Dujun and Zheng, Wenzhao and Zhang, Youmin and Chen, Long},
  journal={arXiv preprint arXiv:2406.19833},
  year={2024}
}
```

OpenStereo: https://github.com/XiandaGuo/OpenStereo
