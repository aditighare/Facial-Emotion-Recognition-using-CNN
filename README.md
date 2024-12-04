# Facial-Emotion-Recognition-using-CNN
--Code for pre-processing and training the model available in 'facial-emotion-recognition.ipynb' file.

--Trained model is saved in 'model.h5'

--To run saved model on new data to recognize emotions:
1) for image data - run 'image_run.py'
2) for real time emotion recognition using web-cam - run 'test_video.py'

# Implementattion Details
Facial Emotion Recognition System trained on fer-2013 dataset using 2 different models of CNN which recognizes the 7 emotions such as angry, disgust, fear, happy, sad, surprise, neutral. The overall accuracy of predicting the emotions correctly is 66.86% for the best model out 2 different implemented models of CNN. The best model has 4 convolution blocks each consisting convolution, max-pooling, ReLu activation and dropout after that flattening is done and further 2 fully connected layers are used with softmax activation function and Adam optimizer. Finally early stopping is used along with reduce learning rate  to increase the overall performance of the model.
(Only best model performing model is uploaded here)
