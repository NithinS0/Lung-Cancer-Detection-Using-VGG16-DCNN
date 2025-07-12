  🧠 Lung Cancer Detection Using VGG16 (DCNN)
  This project implements a Deep Convolutional Neural Network (DCNN) using the pre-trained VGG16 model for lung cancer image classification. It leverages transfer learning to fine-tune VGG16 on a medical imaging dataset, improving accuracy in     detecting various types of lung cancer.
  
  📌 Overview
  Model: VGG16 with custom dense layers
  
  Task: Multi-class classification of lung cancer images
  
  Technique: Transfer learning + data augmentation + fine-tuning
  
  Input size: 128x128 RGB images
  
  Framework: TensorFlow/Keras
  
  📂 Dataset
  Directory structured dataset with subfolders for each class.
  
  Loaded using flow_from_dataframe() for flexible preprocessing.
  
  Train/Validation/Test split: 60% / 20% / 20% (stratified).
  
  🔍 Key Features
  ✅ Transfer learning with pre-trained VGG16 on ImageNet
  
  ✅ Custom classification head with dropout, batch normalization, and L2 regularization
  
  ✅ Data augmentation for better generalization
  
  ✅ Early stopping and learning rate scheduling
  
  ✅ Final model saved as .h5 for future inference
  
  ✅ Accuracy and loss plots for performance tracking
  
  🛠️ Tech Stack
  Python 3.x
  
  TensorFlow 2.x / Keras
  
  Pandas, NumPy
  
  Matplotlib
  
  Scikit-learn
  
  🚀 Training & Evaluation
  bash
  Copy
  Edit
  # Install dependencies
  pip install -r requirements.txt
  
  # Train the model
  python train.py
  
  # Evaluate on test set
  python evaluate.py
  Model achieves an accuracy of ~XX% on the test set (update with your actual result).
  
  📊 Output Visuals
  Training vs Validation Accuracy
  
  Training vs Validation Loss
  
  

  
