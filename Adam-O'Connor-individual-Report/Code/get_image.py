from quick_load import *
from matplotlib import pyplot as plt
import random

# Suppose you have these already:
# mlb = MultiLabelBinarizer()
# class_names = sorted(set(','.join(df['target']).split(',')))
# class_names += ['none']

# Pick a single random index
idx = random.randint(0, len(train_set) - 1)

# Create a 2x3 figure (2 rows, 3 columns)
fig, axs = plt.subplots(2, 3, figsize=(15, 8))  # Adjust size to fit 6 images nicely

axs = axs.flatten()  # Flatten the 2D array of axes into 1D for easy looping

for i in range(6):
    img, label = train_set[idx]  # Call train_set[idx] again each time (new augmentation)
    img = img.permute(1, 2, 0).numpy()  # Convert to HWC format for plotting

    # Convert label tensor to class names
    label = label.numpy()
    label_indices = label.nonzero()[0]  # Get indices where label==1
    label_text = ', '.join([class_names[j] for j in label_indices])

    axs[i].imshow(img)
    axs[i].set_title(f"Label: {label_text}", fontsize=8)
    axs[i].axis('off')

plt.tight_layout()
plt.savefig('view_image.png')
plt.show()
