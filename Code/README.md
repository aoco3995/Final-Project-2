# Code Directory README

This folder contains the core source code for Final Project Team 2.

## Structure

```
Code/
├── train_model.py           # Starts training the model and saves results in training_logs/
├── config.py                # Contains hyperparameters; edit this file to change training settings
├── Unlabeled_test_MARS.py   # Uses a trained model to label unlabeled images and outputs pseudo-labels
├── View_images.py           # Allows viewing of images with a specific label from pseudo-labels output
├── model.py                 # Defines the CNN model architecture used for training and inference
├── quick_load.py            # Loads dataset and displays augmented training image sample
```

## How to Run

1. Clone the repository (if you haven't already):

   ```bash
   git clone [your-repo-url]
   cd [your-repo-name]/Code
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:

   ```bash
   python train_model.py
   ```

   This will begin training the model and save the trained model along with metrics in the `training_logs/` folder.

4. Generate pseudo-labels for unlabeled data:

   ```bash
   python Unlabeled_test_MARS.py
   ```

   This script uses a trained model to automatically label images in the `unlabel_aircraft_fuselage/` directory.
   By default, it uses the final model trained in this project:

   ```python
   model_path = '../Code/Saved_logs/Final_model/model_MARS.pt'
   ```

   To use a different model, modify the `model_path` variable on line 217.

   After execution, a file named `pseudo_labels_MARS.csv` will be generated containing predicted labels for all unlabeled images.

5. View labeled images by target class:

   ```bash
   python View_images.py
   ```

   This module loads the `pseudo_labels_MARS.csv` file and displays randomly selected images that match a given label.
   You can customize the input data and target label by modifying the following variables at the top of the script:

   ```python
   DATA_DIR = "Dataset/Aircraft_Fuselage_DET2023/unlabel_aircraft_fuselage"
   csv_path = "pseudo_labels_MARS.csv"
   TARGET_LABEL = "paint peel".lower()
   ```

   When run, the script opens images one at a time that match the specified label, and saves each image preview as `labeled.jpg`.
   Click 'next' in the interface to view another randomly chosen image with that label.

6. Load and preview training data samples:

   First launch an interactive Python session:

   ```bash
   python
   ```

   Then run:

   ```python
   from quick_load import *
   view_image(0)  # Replace 0 with any index to view a different sample
   ```

   This will load a sample from the training dataset, apply augmentations and masking, and save a visualization to `sample_image.png`.
   Running the same index multiple times will result in different augmentations and mask patterns.

   **Note:** In order to use `view_image()` successfully, you must first uncomment the return values for `original_image` and `row['id']` on line 165 of `dataset_loader.py`:

   ```python
   return image, torch.FloatTensor(target), original_image, row['id']
   ```

   This line is commented out by default:

   ```python
   return image, torch.FloatTensor(target)#, original_image, row['id']
   ```

   Be sure to re-comment the additional return values before training the model again, as the training code expects only two return values from the dataset.

   The saved image includes annotations showing which labels are applied.

## Configuration

Hyperparameters such as learning rate, batch size, number of epochs, and model architecture can be modified in the `config.py` file.

If your project also uses an external configuration file (e.g., `config.json`, `.env`, or YAML):

* Make sure the configuration file is properly set up.
* You can specify a config path like so:

  ```bash
  python train_model.py --config configs/your_config.json
  ```

## Requirements

* Python 3.x
* Packages listed in `requirements.txt`

## Dependencies

This project requires the following Python packages. Most of these can be installed via pip:

```bash
pip install torch torchvision pandas matplotlib seaborn scikit-learn tqdm pillow
```

**Core Libraries:**

* `torch`, `torchvision` – Deep learning and dataset utilities
* `pandas`, `numpy` – Data manipulation
* `matplotlib`, `seaborn` – Plotting and visualization
* `scikit-learn` – Evaluation metrics
* `tqdm` – Progress bars
* `PIL` – Image handling
* `opencv-python` (`cv2`) – Image I/O and annotation

**Standard Library Modules Used:**

* `os`, `json`, `csv`, `random`, `math`, `datetime`, `inspect`, `shutil`, `importlib`

## Notes

* Output files are saved to the `output/` directory.
* Logs and intermediate files go in `logs/`.
* Training logs and model checkpoints are saved in the `training_logs/` directory.
* Pseudo-labeled outputs are saved to `pseudo_labels_MARS.csv` after running `Unlabeled_test_MARS.py`.
* Labeled image previews are saved to `labeled.jpg` after running `View_images.py`.
* Augmented training sample images are saved to `sample_image.png` after running `quick_load.py`.
