# NLP Transformer
> Part of a Student Project at TU Berlin  
> By: [Florian Peters](https://github.com/flpeters)  
> last change: 13.03.2019  


# Architecture
This is an implementation of the __Transformer__ architecture, as described in the 2017 Paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).  

# Tokenization
We are using the __Byte Pair Encoding__ (BPE) Scheme as described in ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019), bettern known as GPT-2.  
The BPE implementation is taken directly from https://github.com/huggingface/, however it seems that the [original link](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization_gpt2.py) we used is no longer valid.

# Code
All the code needed to run the Model is in __Main.ipynb__, with only the Tokenizer / Encoder / Decoder seperated into __textencoder.py__.  
We are using the __fast.ai__ deep learning library, which is based on __pytorch__.  
Originally we were inspired to use this model by ["The Annotated Transformer"](http://nlp.seas.harvard.edu/2018/04/03/attention.html) (2018).  
The three step training process described below was inspired by ["Universal Language Model Fine-tuning for Text Classification"](https://arxiv.org/abs/1801.06146) (2018), and ["Improving Language Understanding by Generative Pre-Training"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (2018).

# How To Use:

- Open the jupyter notebook __Main.ipynb__.
- You should be able to run all the cells up to __Step 1: LM (wiki103)__.
- From here on, there are three steps needed to complete the training of the model (or less, depending on what you want the model to do).
1. __Step 1: LM (wiki103)__: Training a language model for predicting the next word in a sentence on the wikitext103 dataset.
2. __Step 2: LM (dstc7)__: Taking the pretrained model from step one and fine tuning it with the same task on the dstc7 dataset.
3. __Step 3: CL (dstc7)__  Taking the pretrained model from step two and fine tuning it on a classification task on the dstc7 dataset. The classification task we trained on is Subtask nr. 4 from the ["NOESIS: Noetic End-to-End Response Selection Challenge"](http://workshop.colips.org/dstc7/proposals/Track%201%20Merged%20Challenge%20Extended%20Desscription_v2.pdf) (2018).

- All the data is available as a custom data set downloadable from Kaggle [here](https://www.kaggle.com/flpeters/dstctransformer). 
To run the Model, only __data.7z__ is needed. The other files are the raw sources of text, in case you want to try a different text representation.
- Download the data and change the defined paths at the top of each step to point to the correct location within the __data__ folder. If you extract __data.7z__ to the same directory as __Main.ipynb__, the paths should work as they are.
- At the end of each training step, use `learner.save('file_name')` to save the weights to disk. The directory used will be __data/models__. The __data/__ part of the path is equal to whatever you specify in the `path` variable passed to the databunch, and the __models__ part is a new folder that will be created if it doesn't already exist. The file name will be whatever you pass to the function. Be careful though, as using the same filename twice results in the old file being overwritten.
- At the start of each Step you can specify a `pretrained_path`, which should point to the weights saved in the previous step.
- Be aware that when changing from step 2 to step 3, the task changes, and so does the architecture of the Model. When creating the learner, the weights of the pretrained file are converted to fit the new architecture. Saving and restarting on this step might not work properly though.
- After each step, you get a fully working model. During step 1 and 2 you can use the `predict()` function to make the model generate text from an arbitrary input primer. The last step (classification) never worked as intended though, and we didn't get around to fixing it. We will leave it to the reader to improve on what has already been done.
