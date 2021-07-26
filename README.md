# Image-Caption-Attention
Upload your image to get a caption
Please note that the jupyter notebook (ipynb) file will not display correctly in the github preview, please download in order to view it


<div class="alert alert-block alert-success">
<a id='Project Writeup'><h3><font color ='black'>Project Writeup</font></h3></a>

<h2><b>Understanding the entire workflow of Image Captioning using attention model.</b></h2>
<font color='black'>
<h3><b>Step 1:</b></h3>  Creating a pandas dataframe of the token.txt containing the image ids and related tokens. We add start and end tokens to all the captions. All the image paths and captions are stored as a .pkl file for serializing.
    
<h3><b>Step 2:</b></h3> Tokenize the captions. We need to convert all our captions to machine readable numeric data, similar to one-hot encoding. We add zero padding to the end of all captions having a length less than the maximum length of all captions. This helps in creating the caption vocabulary

<h3><b>Step 3:</b></h3>  We preprocess our images by passing our previously created image path through a load image function that transforms the image into a form readable by the inception v3 network (for creating feature maps). We create tensorflow batch object that has 64 training images in each batch.

<h3><b>Step 4:</b></h3> Create a image feature extraction model that takes the 2nd last layer of the inception-v3 having pre-trained weights to convert our  images into 8x8x2048 feature vector. We will be passing our image paths to the pretrained model to extract the feature map and store it as a .npy file in the image folder to be used later.

<h3><b>Step 5</b></h3> We will create training and testing dataset by spliting all our image paths and captions in 80-20 ratio. A  data generator function will read the image path and load the corresponding .npy image feature. The generated dataset is in the form of a tensor dataset which is optimized by autotuned prefetch and shuffling. Every time our model calls for data ,a batch of 64 training images and the corresponding captions are called by utilizing prefetching to reduce idling.

<h3><b>Step 6</b></h3> We define our model by creating encoder , decoder and attention classes. We utilize the tensorflow subclassing api and create our custom training loops. The models inherit from keras.Model parent class.

Encoder: Reads the image feature vector (inception v3 output vector)that has a depth of 2048 and compresses it into 
256 dimensions (same as the embedding dimension) to be fed into the decoder. The attention class creates the attention object that converts the feature vector at each time step of the decoder into a context vector. A context vector helps the model to assign more weights to relevant portions of the feature vector for each caption word. This preserves the spatial information over long captions. Bahanadau attention (Global attention) is implemented;where all the pixels are considered. 

Decoder: Decoder class is used to create the decoder object that uses GRU RNN in its architecture.It takes at each time step an input token, the input feature vector and creates a context vector using the previous time step hidden state and the feature vector and applying the tanh activation. The LSTM takes the context vector concatenated with the actual input token at the previous time step (teacher forcing principle during training) and generates an output state and hidden state. This loop continues till an end token is generated or it has reached the max caption length. The LSTM output is passed through a fully connected layer and softmax layer at each time step to generate probability scores for the entire vocabulary length. 

<h3><b>Step 7:</b></h3> Generating captions: Greedy search or Beam search algorithms are used to select the correct caption word at each time step. In greedy search the vocabulary word with the highest probability is chosen at each time step. The disadvantage of this method is that if the first predicted word is incorrect; all the following generated caption words will be incorrect. This method is computationaly inexpensive. Beam Search utilizes breadth first seacrh (BFS) principles and chooses 'k' predicted words with high probabilities according to 'k' beam width. Using these two selected words it keeps choosing groups of words with high probabilities and removes the rest. This method is computationally expensive but generally produces better results.

<h3><b>Step 8:</b></h3>Bleu Score or bilingual evaluation understudy score is a technique where the generated caption is compared with the actual token in n-grams. The Bleu score has values between 0-1. A score closer to 1 means the caption is very close to the real caption.

<h3><b>Step 9:</b></h3> Gtts (Google text to speech) library has been used to convert the generated text captions into audio. FInally I have build a flask app and deployed it in Heroku (Paas) where the client can upload any image and the backend flask framework will use the trained model to generate captions. The model loads the trained weights of the encoder-decoder and uses the previously created vocabulary to generate the best suited captions. The

<h3><b>Step 10:</b></h3>Once the model is trained the evaluate function will be used to generate the captions for any user image.  The function converts the user image into  inception v3 compatible form by using 'load image function' and  the feature vector output from the the inception model using 'extraction_model' function. This is fed to the encoder to create the feature vector needed by the decoder at every time step. The decoder is given an input start token at the beginning and a loop is run till the max length of the trained caption length. At each step a context vector is generated by the attention unit which focuses on certain parts of the input feature vector (therefore on the input image). This is determined by the already trained attention weights. This context vector is concatenated with the previous time step output word and is fed to the GRU layer which outputs  softmaxed probabilities of all the words in the vocab. Either greedy or beam search is used to chose the best possible word at a time step. This keeps on going till an end token is generated or the max length is reached
    </font>
[Index](#Index): go back to index
</div>
