# Image-Captioning-using-a-dataset
This project builds an image captioning model that generates textual descriptions for images using a CNN + LSTM architecture, trained on the Flickr8k dataset.

# Model Architecture
 **Feature extractor**: Pre-trained CNN (e.g., InceptionV3, ResNet50) extracts image features.
 
 **Captioning model**: An LSTM-based decoder generates text based on image features and word embeddings.
 
 **Loss Function**: Categorical Cross-Entropy.
 
 **Optimizer**: Adam.

# Dataset 
📂**Flicker8k_Dataset**

This model contains about 8,000 images in .jpg, .png format with the captions writen in about 4-5 different ways.

The dataset contains the captions along with the images to be used to train the model to generate the captions for a given image.

# Requirements 

You will need to install the libraries in requirements.txt file. 

# Dataset overview

📂**Flicker8k_Dataset**

This foder contains the images to help model train.

📂**Flicker8k_text**

**Flickr8k.token.txt**	
This folder contains all images captions (5 per images).

**Flickr_8k.trainImages.txt**

Contains filenames for the training set.

**Flickr_8k.testImages.txt**

Contains filenames for the test set.

**Flickr_8k.devImages.txt**

Contains filenames for the validate set.

**Models**

Contains saved models checkpoints (will be created by function define_model)

**descriptions.txt**

Contains the cleaned and preprocessed captions for each image in the format

**tokenier.p**

Stores the Tokenizer object used to convert words into integer sequences and vice versa.

**features.p**

A Pickle file storing precomputed image features extracted from Xception/InceptionV3.
Used during model training to avoid recomputing image features.

**model_checkpoint.keras**

Checkpoint of model saved during traning the model.

**model_final.keras**

Final trained model after all epochs.


# Output example
**Input**

![Testing_Image](https://github.com/user-attachments/assets/02c8cdc5-718c-4e2f-adda-30cafa83813e)

**Output**

![output](https://github.com/user-attachments/assets/2aefa43a-dd2b-44f8-84c5-6eb4943d83f1)


