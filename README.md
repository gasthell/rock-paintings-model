# Rock Paint Segmentation Project

This repository contains code and notebooks for unsupervised and deep learning-based segmentation of petroglyphs and rock art images. The project includes data augmentation, unsupervised segmentation (K-means), and DINOv2-based segmentation pipelines.

## Project Structure
- `dino.ipynb`: DINOv2-based segmentation and training pipeline.
- `unsupervised.ipynb`: Unsupervised segmentation using K-means and DiffCut.
- `notebook-data-augmentation (2).ipynb`: Data augmentation and inpainting for dataset preparation.
- `augmented_data/`: Folder containing augmented images.
- `pre_segmented_results/`: Folder with pre-segmented results.
- `dino_masks/`: Folder with predicted masks from the DINOv2 model.
- `dataset/`: Contains `images/` and `masks/` for training.

## Requirements
See `requirements.txt` for dependencies. Install with:

```
pip install -r requirements.txt
```

## Usage
Open the notebooks in VS Code or Jupyter and follow the cell instructions. Adjust paths as needed for your environment.

## Notes
- Some scripts require CUDA for acceleration.
- Pretrained weights for DINOv2 are downloaded automatically.
- Data augmentation uses PatchMatch for inpainting.

## License
This project is for research and educational purposes.
