# Semi-supervised Change Detection for Cartosat-3 Imagery

This repository contains a PyTorch-based implementation for semi-supervised change detection, adapted for use with Cartosat-3 satellite imagery. The core methodology is based on the paper: **"Revisiting Consistency Regularization for Semi-supervised Change Detection in Remote Sensing Images"**.

This project provides a framework for:
- Preprocessing Cartosat-3 data.
- Training a semi-supervised change detection model.
- Running inference to generate change masks.

## :speech_balloon: Requirements

This repo was tested with `Window 10`, `Python 3.12.10`, `PyTorch  2.8.0`, and `CUDA 12.7`. However, it should be runnable with recent PyTorch versions (>=1.1.0).

The required packages are listed in `requirements.txt`.

### Installation

1.  Create and activate a python environment:
    ```bash
    python -m venv "venvname"
    python/Scripts/activate
    ```

2.  Install the required packages:
    ```bash
    pip3 install -r requirements.txt
    ```

## :speech_balloon: Dataset and Preprocessing

This project is designed to work with Cartosat-3 satellite imagery. Before training the model, the raw data needs to be preprocessed. The preprocessing pipeline involves several steps, executed by scripts in the `utils/` directory.

**Important:** Some of the preprocessing scripts contain hardcoded paths. You may need to modify them to fit your directory structure.

### Preprocessing Workflow

1.  **Place Raw Data:**
    *   Organize your raw Cartosat-3 images. A suggested structure is to have separate folders for the two time points (e.g., `T1` and `T2`) and for the masks.

2.  **Remove XML Files (if any):**
    *   The `utils/xml_remove.py` script deletes `.xml` files from the `mask` directory. If your dataset contains these files, run the script.
    ```bash
    python utils/xml_remove.py
    ```

3.  **Rename Files:**
    *   The `utils/file_change.py` script renames files to a consistent `{id}.tif` format. This script is hardcoded to work with a `test_enaroch` directory. You will need to adapt the paths in this script to match your data structure.

4.  **Convert TIFF to PNG:**
    *   The project uses PNG images for training. The following scripts convert TIFF images to PNGs. These scripts are configurable. You should open them and set the `INPUT_DIR` and `OUTPUT_DIR` variables.
    *   **For RGB images:**
        ```bash
        python utils/tif_topng.py
        ```
    *   **For mask images:**
        ```bash
        python utils/mask_png.py
        ```
        *(Note: This script converts masks to binary PNGs.)*

### Final Data Structure

After preprocessing, your data should be organized in a directory structure that the dataloader can understand. The training configuration in `configs/config_carto.json` expects the following structure:

```
<your_data_directory>/
├── A/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── B/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── label/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── list/
    ├── train_supervised.txt
    ├── 100_train_supervised.txt
    ├──100_train_unsupervised.txt
    ├── train_unsupervised.txt
    ├── test.txt
    └── val.txt
```

*   `A` and `B` folders contain the image pairs from the two time points. A is for past B is for present
*   `label` folder contains the ground truth change masks.
*   `list` folder contains text files that define the data splits for training and validation.

Finally, you must update the `data_dir` path in `configs/config_carto.json` to point to `<your_data_directory>`.

## :speech_balloon: Training

To train the model, you will use the `train.py` script with a configuration file. The main configuration for the Cartosat-3 dataset is `configs/config_carto.json`.

### :point_right: Start Training

To start training with the default configuration, run the following command:

```bash
python train.py --config configs/config_carto.json
```

### :point_right: Configuration

The training process is controlled by the parameters in the `configs/config_carto.json` file. Here are some of the most important parameters you might want to modify:

*   `"name"`: The name of the experiment.
*   `"sup_percent"`: The percentage of labeled data to use for supervised training.
*   `"unsup_percent"`: The percentage of unlabeled data to use for unsupervised training.
*   `"unsupervised_w"`: The weight of the unsupervised consistency loss.
*   `"epochs"`: The total number of training epochs.
*   `"batch_size"`: The batch size for training.
*   `"lr"`: The learning rate.
*   `"save_dir"` and `"log_dir"`: The directories where checkpoints and logs will be saved.

For example, to train a semi-supervised model with 100% labeled data, you would set `"sup_percent": 100` and ensure `"semi": true` in the `model` section of the config file. To train a fully supervised model, you would set `"model": { "supervised": true, "semi": false }`.

### :point_right: Resuming Training

To resume training from a saved checkpoint, use the `--resume` flag:

```bash
python train.py --config configs/config_carto.json --resume <path_to_checkpoint.pth>
```

### :point_right: Monitoring Training

The training progress, including loss and metrics, is logged and can be visualized using TensorBoard. To launch TensorBoard, run:

```bash
tensorboard --logdir saved/Cartosat
```

The log files and model checkpoints will be saved in the directory specified by `save_dir` in your config file (e.g., `saved/Cartosat/`).

## :speech_balloon: Inference

To run inference and generate change masks for a dataset, use the `inference.py` script. You will need a trained model checkpoint.

### :point_right: Run Inference

```bash
python inference.py --config configs/config_carto.json --model <path_to_best_model.pth> --Dataset_Path <path_to_your_dataset> --save
```

### :point_right: Command-Line Arguments

*   `--config`: Path to the config file used during training (e.g., `configs/config_carto.json`).
*   `--model`: Path to the trained model checkpoint (`.pth` file).
*   `--Dataset_Path`: Path to the dataset you want to run inference on. This directory should have the same structure as the training data.
*   `--save`: A flag to save the output prediction masks.

The predicted change masks will be saved as PNG images in the `outputs/<experim_name>/` directory, where `<experim_name>` is specified in your config file. An HTML file with the evaluation metrics will also be generated in the same directory.

## :speech_balloon: Pre-trained Models

*You can add links to your pre-trained models here.*

## :speech_balloon: Citation

If you find this project useful for your research, please consider citing the original paper:

```
@misc{bandara2022revisiting,
      title={Revisiting Consistency Regularization for Semi-supervised Change Detection in Remote Sensing Images},
      author={Wele Gedara Chaminda Bandara and Vishal M. Patel},
      year={2022},
      eprint={2204.08454},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## :speech_balloon: Acknowledgements

This code is based on the official implementation of [CCT](https://github.com/yassouali/CCT). The code structure was based on the [Pytorch-Template](https://github.com/victoresque/pytorch-template/blob/master/README.m), and the ResNet backbone was downloaded from [torchcv](https://github.com/donnyyou/torchcv).
