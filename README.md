# Autonomous Driving Semantic Segmentation

This project implements semantic segmentation on natural environment driving scenes using a customized ResNet-based model, built on a subset of the WildScenes dataset.

## 📘 Project Overview

- **Dataset**: Subset of WildScenes (CSIRO 2023)
- **Model**: Custom CNN based on ResNet
- **Loss Function**: Dice Loss + Cross Entropy
- **Metrics**: Mean Intersection over Union (mIoU)

## 📁 Dataset & Model Download

Due to large file sizes, the following components are not hosted in this repository:

- `dataset.zip` (8.08GB): A preprocessed subset of the original 103.15GB WildScenes benchmark, containing selected scenes and labels used for training.
- `label_set.zip` (60.5MB): Processed label files used during model training and evaluation.
- `resnet_bestmodel.pt` (714MB): Pretrained model checkpoint with the best performance on the validation set.

📩 Please contact the author if access is needed.

## 🔍 Key Files

- `notebooks/finally.ipynb`: Main training & evaluation notebook.
- `notebooks/analy_label.ipynb`: Label distribution and visualization.
- `utils/showpredict.py`: Inference and result display script (GUI with Tkinter).
- `docs/report_final.pdf`: Final project report.
- `docs/COMP9517_24T2_Group_Project_Specification.pdf`: Original project specifications.
- `analysis/*.csv`: Label frequency and image analysis results.
- `utils/data_prepro.drawio.png`: Data preprocessing flowchart.

## 🔬 Dataset Source

- **WildScenes Benchmark**  
  CSIRO Robotics  
  [https://csiro-robotics.github.io/WildScenes/](https://csiro-robotics.github.io/WildScenes/)

## 🧠 Summary

This project demonstrates effective preprocessing, training, and evaluation strategies for semantic segmentation in outdoor autonomous driving contexts, achieving balanced accuracy across multiple terrain and object classes.

A GUI tool is also provided for visualizing predictions on test images using the trained model.

## 🧑‍💻 Authors
- Po-Hsun Chang 📧 chris89019@gmail.com  
- Group project with: Muxin Qiao, Jingwen Yang, Xuewei Zhuang, Tianyi Dou
