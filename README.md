# ASL Alphabet Recognition - MATLAB

![CNN Performance](assets/accuracy_plot.png) 

## Key Features
- Compared **KNN (82%)**, **SVM (88%)**, and **CNN (95%)** models
- Full preprocessing pipeline (resizing, normalization)
- Interactive confusion matrix visualization

## Technical Stack
```matlab
% Sample code snippet (from your cnn_knn_svm.m)
layers = [
    imageInputLayer([64 64 3])
    convolution2dLayer(3, 32, 'Padding','same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride',2)
    fullyConnectedLayer(29)
    softmaxLayer()
    classificationLayer()
];
