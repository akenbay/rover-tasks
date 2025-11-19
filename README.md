# rover-tasks

This repository contains solutions to tasks from the Rover recruitment process. Each task is organized in its own folder.

## Task 1: Football Elements Detection

**Location:** `1_task/`

### Objective
Classify and detect football players, ball, and referee in images using object detection.

### Dataset
- **Source:** [Futbol Players Dataset](https://universe.roboflow.com/ilyes-talbi-ptwsp/futbol-players) from Roboflow Universe
- **Classes:** Football players, ball, referee

### Methodology

1. **Data Augmentation**
   - Used the `albumentations` package for data augmentation
   - Created 3 different augmented variants for each image

2. **Model Training**
   - Trained two YOLO models on Google Colab:
     - Model 1: Trained on **augmented dataset**
     - Model 2: Trained on **original dataset**
   
3. **Results Organization**
   - `1_task/aug/` - Contains model and results trained on augmented data
   - `1_task/og/` - Contains model and results trained on original data

### Testing the Models

**Test model trained on augmented dataset:**
```bash
yolo detect predict model=1_task/aug/best.pt source=1_task/dataset_football/test/images
```

**Test model trained on original dataset:**
```bash
yolo detect predict model=1_task/og/best.pt source=1_task/dataset_football/test/images
```

### Results
Results will be saved in `runs/detect/predict/` with annotated images showing detected players, ball, and referee with bounding boxes and confidence scores.

In comparison, model A trained on augmented dataset showed similar but very volatile results to model B trained on original dataset. While model A showed variance, it might was too much and was not suited for this task. Model A struggled to generelize. 

---