# 🖖 ASL Alphabet Recognition: MATLAB Multi-Algorithm Study

![ASL Classification Demo](assets/accuracy_comparison.png)  
*Comparative performance of KNN, SVM, and CNN models*

## 🔍 Project Overview
**Goal**: Develop and compare machine learning approaches for American Sign Language alphabet recognition  
**Impact**: Bridges communication gap between deaf and hearing communities  
**Dataset**:  
- 29 classes (A-Z + space/nothing/del)  
- Originally 3,000 images/class → Downsampled to 500/class for testing  
- Image format: 200×200 RGB → Preprocessed to 64×64 grayscale  

## 🏆 Key Findings (Full Dataset)
| Model | Accuracy | Error Rate | Training Time | Strengths |
|-------|----------|------------|---------------|-----------|
| **KNN** | 99.97% | 0.03% | 2 min | Best overall performer |
| **CNN** | 99.95% | 0.05% | 1.5 hr | Handles complex patterns |
| **SVM** | 99.68% | 0.32% | 15 min | Balanced performance |

## 📊 Detailed Results
### KNN (k=3)
- **99.9773% test accuracy**  
- Near-perfect classification (only 0.03% errors)  
- Letters L (Recall: 0.995) and O (Precision: 0.995) showed minimal drops  

### SVM (RBF Kernel)
- **99.6818% test accuracy**  
- ECOC framework for multiclass  
- Occasional misclassifications (E/O confusion)  

### CNN (Custom Architecture)
- **99.9545% test accuracy**  
- 3 convolutional blocks with BatchNorm/ReLU  
- Minor challenges with letters G (Recall: 0.99) and H (Precision: 0.99)  

## 🛠️ Technical Implementation
```matlab
% Data Splitting (60/20/20)
[trainData, tempData] = splitEachLabel(imds, 0.6, 'randomized');
[valData, testData] = splitEachLabel(tempData, 0.5, 'randomized');

% CNN Architecture
layers = [
    imageInputLayer([64 64 3])
    convolution2dLayer(3, 32, 'Padding','same')
    batchNormalizationLayer()
    reluLayer()
    maxPooling2dLayer(2, 'Stride',2) 
    fullyConnectedLayer(256)
    dropoutLayer(0.5)
    fullyConnectedLayer(29)
    softmaxLayer()
    classificationLayer()
];
