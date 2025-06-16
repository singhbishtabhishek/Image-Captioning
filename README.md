# Image-Captioning-using-a-dataset
This project builds an image captioning model that generates textual descriptions for images using a CNN + LSTM architecture, trained on the Flickr8k dataset.

# Model Architecture
 **Feature extractor**: Pre-trained CNN (e.g., InceptionV3, ResNet50) extracts image features.
 
 **Captioning model**: An LSTM-based decoder generates text based on image features and word embeddings.
 
 **Loss Function**: Categorical Cross-Entropy.
 
 **Optimizer**: Adam.

# Dataset 
**Flicker8k_Dataset**

This model contains about 8,000 images in .jpg, .png format.

# Requirements 

You will need to install the libraries in requirements.txt file. 

# Output example
**Input**

![Testing_Image](https://github.com/user-attachments/assets/02c8cdc5-718c-4e2f-adda-30cafa83813e)

**Output**

![output](https://github.com/user-attachments/assets/2aefa43a-dd2b-44f8-84c5-6eb4943d83f1)


