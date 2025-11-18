# SewerMan

This is the official implementation of the paper **"SewerMan: A Vision-Language Collaborative Framework with Domain Prior for Robust Multi-Label Sewer Defect Classification"**.

## 🚀 Overview

SewerMan is a novel framework that synergizes a visual-language collaboration model with advanced data pre-processing for robust multi-label sewer defect classification. 
It effectively integrates domain priors (like sewer material information) with cross-modal interactions to achieve state-of-the-art performance on the Sewer-ML and QV-Pipe dataset.

## 📊 Datasets and Resources

The experiments in the paper are conducted on the **Sewer-ML** and **QV-Pipe** datasets.

### 1. Material Labels & Pre-trained Weights

**The following crucial resource files are too large to host directly on GitHub. Please download them from our Baidu NetDisk (百度网盘) repository:**

- **Link:** https://pan.baidu.com/s/1FDBrb40Q1hKOqwLg_eVHzw
- **Access Code/Extraction Code (提取码):** st6m


#### **Files to Download:**
﻿
After downloading, please place them in the corresponding directories in your local project.
﻿
1.  **Material Label Files (Place in `Resources/` directory):**
    - sewer_material_labels_part1.csv and sewer_material_labels_part2: Automatically generated material labels for the Sewer-ML dataset.
    - qvpipe_material_labels.csv: Automatically generated material labels for the QV-Pipe dataset.
﻿
2.  **Pre-trained Model Weights (Place in `Resources/` directory):**
    - resnet50_stl_material.pth: The pre-trained ResNet-50 model for automatic pipe material classification (used in **Mode 2**).
    - SewerNet_Sewer_ML.ckpt : The final Sewer_ML pre-trained weights of our SewerMan model for inference.
    - SewerNet_QV-Pipe.ckpt : The final QV-Pipe pre-trained weights of our SewerMan model for inference.
  

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/zkkzxk/SewerMan.git
    cd SewerMan
    ```

2.  **Create a conda environment and install dependencies:**
    ```bash
    conda create -n sewerman python=3.9
    conda activate sewerman
    pip install pytorch_lightning =0.9.0
    pip install pytorch =1.8.2
    ...
    ```
    
 # 🏃‍♂️ Usage

### Using Sewer Material Information (Two Modes)

We provide two flexible modes to integrate sewer material information into SewerNet.

#### Mode 1: Using Pre-Provided Labels (For Exact Reproduction)

If you have downloaded the material_labels files, you can load the ground-truth label for each image.

#### Mode 2: Automatic Material Classification (For Practical Deployment - DEFAULT)
This is the mode used for all experiments in the paper. It uses a pre-trained model to automatically predict the material, making the pipeline fully automated.

1.Ensure you have downloaded resnet50_stl_material.pth. 
2.Just run SewerMan/inference.py


📜 Citation
If you find our work useful in your research, please consider citing:
@article{zhong2024sewerman,
  title={SewerMan: A Vision-Language Collaborative Framework with Domain Prior for Robust Multi-Label Sewer Defect Classification},
  author={Zhong, Xuke and Zhong, Yuzhong and Xiao, Quan and Qiang, Hu and You, Xingxing and Dian, Songyi},
  journal={Applied Soft Computing},
  year={2024}
}

📧 Contact
For any questions or issues regarding the code or the paper, please open an issue on GitHub or contact us at [zxk995zkk@163.com].


