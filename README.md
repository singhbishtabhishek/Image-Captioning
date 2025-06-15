# Image-Captioning-using-a-dataset
This project builds an image captioning model that generates textual descriptions for images using a CNN + LSTM architecture, trained on the Flickr8k dataset.

# Model Architecture
 **Feature extractor**: Pre-trained CNN (e.g., InceptionV3, ResNet50) extracts image features.
 
 **Captioning model**: An LSTM-based decoder generates text based on image features and word embeddings.
 
 **Loss Function**: Categorical Cross-Entropy.
 
 **Optimizer**: Adam.
