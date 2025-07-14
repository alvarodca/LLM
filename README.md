# LLM
 Large Language Model from Scratch

This is a project file which follows the reading of Build A Large Language Model from Scratch written by Sebastian Raschka.
The following files are just a summary of the different covered chapters

Files:
Chapter1.ipynb <- Jupyter file with notes on understanding LLMs
Chapter2.ipynb <- Jupyter file with notes on working with text data
Chapter3.ipynb <- Jupyter file with notes on Attention Mechanisms
Chapter4.ipynb <- Jupyter file with notes on implementing the GPT model
Chapter5.ipynb <- Jupyter file with notes on Pretraining on unlabeled data
Chapter6.ipynb <- Jupyter file with notes on Fine-Tuning the model for classification
Chapter7.ipynb <- Jupyter file with notes on Fine-Tuning the model for instructions


AppendixA.ipynb <- Jupyter file with notes on Pytorch basics
AppendixD.ipynb <- Jupyter file with notes on how to improve training loop (done on google collab due to my GPU limitations)
AppendixE.ipynb <- Also required GPU and copying a lot of code, however, the chapter was interesting

chapter02.py, chapter03.py, chapter04.py, chapter05.py <- python file containing a copy of specific functions to implement them easily

model.pth <- First GPT model for inference
model_and_optimizer.pth <- First GPT model for future training
review_classifier.pth <- Classifier model from Chapter 6

gpt_download.py <- File from book to load existing weights

gpt2 <- Loaded trainable weights from GPT2

sms_spam_collection <- contains dataset used for spam classification chapter 6
test.csv, train.csv, validation.csv <- CSV splitted files of the above mentioned folder

instruction-data.json <- json file for dataset Chapter 7

the-verdict.txt <- txt file used on chapter 2 and 5
