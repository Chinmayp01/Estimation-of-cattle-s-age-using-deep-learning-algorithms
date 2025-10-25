# 🧠 Project: A database of dentition images of Indian breed cattle and estimation of cattle’s age using deep learning algorithms

## 🧩 Environment Setup

This project requires a **Python 3.9 environment** with specific library versions to ensure compatibility with TensorFlow `< 2.11` and other dependencies used during model training and inference.

---

### ⚙️ Recommended Versions

| Library | Version | Notes |
|----------|----------|-------|
| Python | 3.9 | Compatible with TensorFlow < 2.11 |
| TensorFlow | < 2.11 | Core deep learning framework |
| NumPy | 1.25.2 | Required version for TensorFlow < 2.11 |
| OpenCV | Compatible with NumPy 1.25.2 | Used for image preprocessing and visualization |
| PyTorch | Latest stable | Vision Transformer models |
| Ultralytics | Latest | Required for YOLOv8 segmentation and detection |
| Matplotlib / Seaborn | Latest | For plotting training and evaluation metrics |
| scikit-learn | Latest | For evaluation utilities and confusion matrix generation |

---

### 🧱 Creating the Virtual Environment

To ensure all dependencies are isolated, it is recommended to use a **virtual environment**.  
You can create and activate it as follows:

#### 🪟 On Windows (using Anaconda)
```bash
# Create a new environment
conda create -n tf_env python=3.9 -y

# Activate the environment
conda activate tf_env
# Install TensorFlow (version < 2.11)
pip install "tensorflow<2.11"

# Install Ultralytics for YOLOv8
pip install ultralytics

# Install PyTorch (with CUDA if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional utilities
pip install matplotlib seaborn scikit-learn pillow tqdm labelme2coco tensorflow_addons pandas 
# Install compatible NumPy and OpenCV
pip install numpy==1.25.2
# recommended
pip install opencv-python==4.8.1.78
```



COMPLETE FOLDER STRUCTURE OVERVIEW
### 📁 Folder Structure Overview
```text
PROJECT_SUBMISSION_EAAI/
│
├── Dataset
│
├── DATASET_WS_VERIFICATION
│
├── GUI
│
├── MODEL_TRAINING
│
└── RESULTS
```


## 🧠 MODEL TRAINING

The `MODEL TRAINING` directory contains all scripts and experiments related to model development.  
This folder includes multiple subfolders, each corresponding to specific experiments and training configurations.

---

### 📁 Folder Structure Overview
```text
MODEL TRAINING/
│
├── 1/ # Experiment 1 – Cattle Dataset with 4 Age Groups
│ ├── Models_Combined.ipynb
│ ├── Models_Combined_EfficientNet.ipynb
│
├── 2/ # Experiment 2 – Cattle Dataset with 7 Age Groups
│ ├── Models_Combined.ipynb
│ ├── Models_Combined_EfficientNet.ipynb
│
├── 3/ # Experiment 3 – Cattle Dataset with 15 Age Groups
│ ├── Models_Combined.ipynb
│ ├── Models_Combined_EfficientNet.ipynb
│
├── Yolo_training/ # YOLO Model Training Scripts (for segmentation and detection)
│ ├── (3 code files for YOLO experiments)
│
└── VisionTransformTraining.ipynb # Vision Transformer model training script
```


---

### ⚙️ Running the Experiments

Each numbered folder (`1`, `2`, `3`) corresponds to an **experiment**:
- **Experiment 1:** 4 Age Groups  
- **Experiment 2:** 7 Age Groups  
- **Experiment 3:** 15 Age Groups  

Each experiment contains **two Jupyter notebooks**:
1. `Models_Combined.ipynb`
2. `Models_Combined_EfficientNet.ipynb`

These notebooks train **8 different models** for the respective dataset configuration.

---

### 🚀 How to Run

1. Open the desired notebook (e.g., `Models_Combined.ipynb` or `Models_Combined_EfficientNet.ipynb`).
2. Click **Run All** to execute all cells sequentially.
3. When prompted by a pop-up window, **select the `Project_Submission_EAAI` folder** as your home directory.
4. Minimize the window if required — the notebook will automatically handle paths and begin training.
5. Training progress, metrics, and saved models will appear in the respective result folders.

---

### 🤖 YOLO Training

The `Yolo_training` folder contains **three code files**, each implementing YOLO-based segmentation/detection tasks.  
Each script is **self-documented**, explaining the functionality of each function in detail.

---

### 🧩 Vision Transformer Training

The `VisionTransformTraining.ipynb` file trains a **Vision Transformer (ViT)** model on the dataset.  
You can execute it directly by running all cells in the notebook.

---

✅ **Tip:** All model weights, training logs, and results are automatically saved to the appropriate directories under `MODEL_RESULTS/` folder.

## 📊 RESULTS

The `RESULTS` directory contains **all essential files, models, and scripts** required to reproduce and verify the outcomes reported in the research paper.  
This folder consolidates **trained models, accuracy logs, result images, and supporting scripts** for seamless validation.

---

### 📁 Folder Structure Overview
```text
RESULTS/
│
├── ACCURACY_CURVES_DATA/ # Contains training and validation logs for plotting learning curves
│ ├── (log files for each experiment)
│
├── MODEL_FILES/ # Stores trained model files (.h5, .pth, .pt)
│ ├── Model_Files_4_Classes/
│ ├── Model_Files_7_Classes/
│ ├── Model_Files_15_Classes/
│ └── best.pt # Best-performing YOLO model
│
├── RESULT_IMAGES/ # Contains all generated result images for analysis and paper verification
│
└── SCRIPTS/ # Python scripts for generating metrics, curves, and other evaluation results
├── (each script name indicates its function)

```
---

### 🧠 Overview

- The **`MODEL_FILES`** folder includes all **trained model weights** used in experiments — available in `.h5`, `.pth`, and `.pt` formats.  


- The **`ACCURACY_CURVES_DATA`** folder stores **log directory files** required for visualizing learning and validation performance curves.  

- The **`RESULT_IMAGES`** folder contains **visual outputs** generated during testing and validation, 
  - Classification results  
  - Comparison matrix visuals 

- The **`SCRIPTS`** folder includes **Python and notebook files** that generate figures, metrics, and tables.  
  Each script is **clearly named** to indicate its purpose (e.g., accuracy plotting, confusion matrix generation, etc.).

---

### 🚀 Usage Instructions

1. Open the desired **script** from the `SCRIPTS` directory.
2. Click **Run All** to execute all cells sequentially.
3. When prompted by a pop-up window, **select the `Project_Submission_EAAI` folder** as your home directory.
4. Minimize the window if required — the notebook will automatically handle paths and begin training.
5. Ensure the paths to model files or log directories are correctly set.
5. Run the script to reproduce:
   - Learning curves  
   - Accuracy and loss metrics  
   - Confusion matrices  
   - Predicted outputs and qualitative results  

---

✅ **Tip:**  
For validation, load the corresponding model weights from `MODEL_FILES/` and the logs from `ACCURACY_CURVES_DATA/` to reproduce the reported figures and tables exactly.


## 🧾 DATASET_WS_VERIFICATION

The `DATASET_WS_VERIFICATION` folder contains **datasets without segmentation masks**, specifically used for **validating the Vision Transformer (ViT)** models — both **with segmentation (ViT-WS)** and **without segmentation (ViT-WOS)**.

---

---

### 🧠 Overview

- These datasets are used for **performance verification** of Vision Transformer models:
  - **ViT-WS (With Segmentation):** Uses pre-segmented input images.
  - **ViT-WOS (Without Segmentation):** Uses original images without segmentation masks.

## 🧾 GUI

The folder is just for reference to create a GUI interface in hugging face. 

## 📚 Citation

If you use this repository, dataset, or any part of this work in your research, please cite the following paper:

> Chinmay Vijay Patil, Ankit Ashokrao Bhurane, Preeti Ghasad, Vipin Kamble,  
> Manish Sharma, Anand Singh, Nareshkumar Nandeshwar, Ru-San Tan, Rajendra Acharya,  
> **A database of dentition images of Indian breed cattle and estimation of cattle’s age using deep learning algorithms**,  
> *Engineering Applications of Artificial Intelligence*, Volume 162, Part D, 2025, 112172,  
> ISSN 0952-1976, [https://doi.org/10.1016/j.engappai.2025.112172](https://doi.org/10.1016/j.engappai.2025.112172).

### 🔹 BibTeX
```bibtex
@article{PATIL2025112172,
  title   = {A database of dentition images of Indian breed cattle and estimation of cattle’s age using deep learning algorithms},
  journal = {Engineering Applications of Artificial Intelligence},
  volume  = {162},
  pages   = {112172},
  year    = {2025},
  issn    = {0952-1976},
  doi     = {https://doi.org/10.1016/j.engappai.2025.112172},
  url     = {https://www.sciencedirect.com/science/article/pii/S0952197625021803},
  author  = {Chinmay Vijay Patil and Ankit Ashokrao Bhurane and Preeti Ghasad and Vipin Kamble and Manish Sharma and Anand Singh and Nareshkumar Nandeshwar and Ru-San Tan and Rajendra Acharya},
  keywords = {Livestock management, Deep learning, Instance segmentation, Convolutional neural networks, Vision transformer, Automated age estimation}
}



