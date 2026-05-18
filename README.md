# ViLSSeg

ViLSSeg is a medical image segmentation repository that integrates visual and language information. The implementation is built using PyTorch and PyTorch Lightning, combining a vision encoder, a text encoder, Mixture-of-Experts output, and an uncertainty mask compensation strategy.

## Key Features

- Vision-language guided medical image segmentation
- Pretrained Transformer-based image and text encoders
- Saves checkpoints for minimum validation loss and maximum Dice score
- Supports prompt-style datasets such as `QaTa`, with multiple dataset path examples in config


## Datasets and Configuration

The default dataset paths in `config/training.yaml` are:

- `./data/QaTa-Covid19-v2/prompt/train.csv`
- `./data/QaTa-Covid19-v2/Train`
- `./data/QaTa-Covid19-v2/prompt/test.csv`
- `./data/QaTa-Covid19-v2/Test`

The config file also contains commented examples for other datasets:

- `MoNuSeg`
- `MosMedDataPlus`

Please adjust dataset paths to match your actual data layout.

> Note: You can refer to [LViT](https://github.com/HUANGLIZI/LViT) to download the dataset.

## Usage

### Training

```bash
python train.py --config ./config/training.yaml
```

The training script will:

- Load the YAML configuration
- Create training and validation datasets
- Initialize `ViLSSeg`
- Train using `pytorch_lightning.Trainer`
- Save checkpoints for minimum `val_loss` and maximum `val_dice`

### Evaluation / Inference

```bash
python evaluate.py --config ./config/test.yaml
```

The evaluation script will:

- Load the model and checkpoint
- Create the test dataset
- Measure inference time per sample
- Compute FLOPs and parameter counts
- Run `trainer.test()` and print test metrics

> The evaluation script currently contains a hardcoded checkpoint path: `xxx/medseg_max_dice.ckpt`. Update this path to your actual checkpoint file before use.

## Configuration Reference

- `TRAIN.train_batch_size`: Training batch size
- `TRAIN.valid_batch_size`: Validation/test batch size
- `TRAIN.lr`: Initial learning rate
- `TRAIN.min_epochs` / `TRAIN.max_epochs`: Training epoch range
- `TRAIN.patience`: Early stopping patience
- `TRAIN.device`: GPU device ID
- `MODEL.bert_type`: Text encoder model path or pretrained name
- `MODEL.vision_type`: Vision encoder model path or pretrained name
- `MODEL.project_dim`: Projection dimension for image/text features
- `DATA.train_csv_path` / `DATA.train_root_path`: Training CSV path and image root path
- `DATA.test_csv_path` / `DATA.test_root_path`: Test CSV path and image root path

## Implementation Notes

- The text encoder parameters are frozen by default and do not train.
- `ViLSSeg` fuses visual and textual features to produce a main segmentation output, an intermediate `os4` output, and an uncertainty mask.
- `engine/wrapper.py` uses `GeneralizedDiceFocalLoss` to combine multiple output layers.
- `train.py` saves two best checkpoints using `ModelCheckpoint`: best `val_loss` and best `val_dice`.

## Acknowledgement
This code is based on [LVIT](https://github.com/HUANGLIZI/LViT), [ViT](https://github.com/google-research/vision_transformer),[LanGuideMedSeg](https://github.com/Junelin2333/LanGuideMedSeg-MICCAI2023) and [CLIP](https://github.com/openai/CLIP) . 

## License

This repository does not include an explicit LICENSE file. Add a license file if you intend to release the project under an open-source license.
