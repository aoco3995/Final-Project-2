import cv2
import csv
import os

# Path to image directory and CSV file
DATA_DIR = "../../Code/Dataset/Aircraft_Fuselage_DET2023/unlabel_aircraft_fuselage"
csv_path = "../../Code/pseudo_labels_MARS.csv"

# Set the label you're interested in (case-insensitive match)
TARGET_LABEL = "paint peel".lower()

# Load and filter CSV entries
with open(csv_path, newline='') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    entries = [
        (img, label)
        for img, label in reader
        if TARGET_LABEL in label.lower()
    ]

total = len(entries)
if total == 0:
    print(f"No images found with label '{TARGET_LABEL}'.")
    exit()

index = 0

while index < total:
    image_name, label = entries[index]
    image_path = os.path.join(DATA_DIR, image_name)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[{index+1}/{total}] Could not read image: {image_path}")
        index += 1
        continue

    labeled_image = image.copy()
    cv2.putText(labeled_image, f"Label: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    output_path = "labeled.jpg"
    cv2.imwrite(output_path, labeled_image)
    print(f"[{index+1}/{total}] Saved image '{image_name}' with label: {label}")

    input("Press Enter to view the next matching image...")
    index += 1

print(f"Finished displaying all images with label: '{TARGET_LABEL}'.")
