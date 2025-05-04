# Adam O'Connor - Individual Code Contributions

This folder contains copies of selected code modules that were specifically worked on by **Adam O'Connor** as part of the individual contribution for the final project.

> **Important:** These scripts are not intended to be run from within this folder. Instead, navigate to the main `Code/` directory and execute all scripts from there to ensure proper handling of file paths and access to all dependencies.

## Included Modules

* `dataset_loader.py` – Loads and preprocesses dataset images and labels. Includes logic for image masking and multi-label handling.
* `get_image.py` – Utility for fetching and working with image files.
* `quick_load.py` – Loads the training and test sets, then visualizes a randomly augmented training image.
* `View_images.py` – Uses the pseudo-labels CSV and unlabeled image data to display images by label category.

## How to Use

1. **Navigate to the main code directory:**

   ```bash
   cd [project-root]/Code
   ```

2. **Launch Python for interactive use of modules like `quick_load.py`:**

   ```bash
   python
   >>> from quick_load import *
   >>> view_image(0)  # Replace 0 with desired sample index
   ```

   This saves the image to `sample_image.png`. The index refers to a row in the dataset, and even with the same index, image masking/augmentation will vary each run.

3. **Reminder about dataset\_loader.py:**
   To use the `view_image()` function from `quick_load.py`, you must temporarily uncomment the return line in `dataset_loader.py`:

   ```python
   return image, torch.Floa
   ```
