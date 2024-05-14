This is a project from the Udemy course Machine Learning AZ - Python - R showcasing Convolutional Neural Networks (CNNs) using TensorFlow

With cnn_train.py you can train and save the model.  
Unfortunately, I cannot upload the datasets but there are a lot of Cats and Dogs datasets online.

The structure for the code to work must be:  
/dataset/training_set/cats  
/dataset/training_set/dogs  
/dataset/test_set/cats  
/dataset/test_set/dogs  
and the filenames must be:  
training_set cats: cat.1.jpg -> cat.4000.jpg  
training_set dogs: dog.1.jpg -> gog.4000.jpg  
test_set cats: cat.4001.jpg -> cat.5000.jpg  
test_set dogs: dog.4001.jpg -> dog.5000.jpg  


With test_cnn.py you can test the model.  
I have created a basic UI with Tkinter.

You can also test it online on a Hugging Face space I have created with Gradio here:  
https://huggingface.co/spaces/eltarny/minimal
