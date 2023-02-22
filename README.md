# DeepWeeds
Classification of different weed species by using convolutional neural networks.

The dataset used in this project is the DeepWeeds dataset:
https://github.com/AlexOlsen/DeepWeeds

A deep network architecture with 7 convolutional layers and 2 fully connected layers with dropout layers in between has been used. After 1 hour and 40 minutes, the trained netowrk showed promising results with a training accuracy of 81%, a validation accuracy of 76%, and a test accuracy of 73%. The following are visual representations of the training process and its results.

Variation of loss during training:

![Loss](https://user-images.githubusercontent.com/65850584/220749356-3a9beaf8-87b9-478d-a75e-35234d145e53.png)


Variation of accuracy during training:

![Accuracy](https://user-images.githubusercontent.com/65850584/220749439-e5b107e9-c267-4c88-9cf3-d5739bce6489.png)


Confusion matrix of test data on the trained network:

![Confusion Matrix](https://user-images.githubusercontent.com/65850584/220749580-b9134c2b-8466-4d8b-bc2d-d9a4e4328935.png)
