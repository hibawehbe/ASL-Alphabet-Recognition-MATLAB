# 🖖 ASL Alphabet Recognition: Multi-Algorithm Comparison (MATLAB)

![ASL Recognition Demo](assets/demo.gif) *<!-- Consider adding a demo GIF -->*

*A comparative study of KNN, SVM, and CNN for American Sign Language recognition*

## 🌟 Key Results (Full Dataset)
| Model | Accuracy | Error Rate | Training Time | Best For |
|-------|----------|------------|---------------|----------|
| **KNN** | 99.97% | 0.03% | ~2 min | Overall best performer |
| **CNN** | 99.95% | 0.05% | ~1.5 hr | High accuracy |
| **SVM** | 99.68% | 0.32% | ~15 min | Balanced performance |

## 📌 Project Highlights
- **Dataset**: 29 classes (A-Z + space/nothing/del) with 500 images each
- **Preprocessing**: 
  - Resizing to 64×64 pixels 
  - Grayscale normalization
  - 60/20/20 train/val/test split
- **Perfect Classification**: Most letters achieved 1.0 precision/recall

## 🛠️ Technical Implementation
```matlab
% CNN Architecture (From your presentation)
layers = [
    imageInputLayer([64 64 3])
    convolution2dLayer(3, 32, 'Padding','same')
    batchNormalizationLayer()
    reluLayer()
    maxPooling2dLayer(2, 'Stride',2)
    dropoutLayer(0.5)
    fullyConnectedLayer(29)
    softmaxLayer()
    classificationLayer()
];
