import cv2
import numpy as np
import os
import albumentations as albu

train_images_og_path="1_task/dataset_football/train/images"
train_labels_og_path="1_task/dataset_football/train/labelTxt"

train_images_augmented_path="1_task/dataset_football/train_augmented/images"
train_labels_augmented_path="1_task/dataset_football/train_augmented/labelTxt"

os.makedirs(train_images_augmented_path,exist_ok=True)
os.makedirs(train_labels_augmented_path,exist_ok=True)

aug_factor=3

# Create augmentation setup
transform = albu.Compose(
    [
        albu.HorizontalFlip(p=0.5),
        albu.Rotate(limit=15, p=0.7),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        albu.RandomGamma(gamma_limit=(80, 120), p=0.3),
        albu.RandomScale(scale_limit=0.1, p=0.5),
    ],
    
    bbox_params=albu.BboxParams(format="yolo", label_fields=['class_labels'])
)


# Read image and labels, transform, and save
for filename in os.listdir(train_images_og_path):
    if filename.endswith(('.jpg')):
        
        #Create paths
        image_path = os.path.join(train_images_og_path, filename)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(train_labels_og_path, label_filename)

        # Read Image
        image = cv2.imread(image_path)
        # Convert BGR to RGB (Albumentations prefers RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        # Read labels
        yolo_boxes = []
        class_labels = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # YOLO format: [class_id x_c y_c w h]
                        class_id = int(parts[0])
                        # Convert the normalized coordinates to floats and append
                        bbox = [float(p) for p in parts[1:]] 
                        yolo_boxes.append(bbox)
                        class_labels.append(class_id)
        except FileNotFoundError:
            print(f"Warning: No label file found for {filename}. Skipping.")
            continue

        # Apply Augmentation and make 3 different copies
        for i in range(aug_factor):
            # Apply the transform
            augmented_data = transform(image=image, bboxes=yolo_boxes, class_labels=class_labels)

            aug_image = augmented_data['image']
            aug_bboxes = augmented_data['bboxes']
            aug_labels = augmented_data['class_labels']

            # New filenames for the augmented data
            new_img_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
            new_label_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.txt"
            
            # Save Augmented Image
            # Convert back to BGR before saving with cv2
            cv2.imwrite(os.path.join(train_images_augmented_path, new_img_filename), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

            # Save Augmented Labels
            with open(os.path.join(train_labels_augmented_path, new_label_filename), 'w') as f:
                # Go through two arrays at one time
                for bbox, label in zip(aug_bboxes, aug_labels):
                    
                    # Make a Yolo line
                    line = f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                    f.write(line)

print("Augmentation complete!")
