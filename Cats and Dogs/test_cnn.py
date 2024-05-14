from pathlib import Path

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model

current_path = Path.cwd()
# Load the saved model from the file
cnn = load_model(current_path / 'models' / 'catsanddogsmodel.keras')

image_path = ''
def test_image(test_file):
    # Load the image as numbers
    test_image = image.load_img(test_file, target_size = (64, 64))
    
    # Make it an np.array and Feature Scale
    test_image = image.img_to_array(test_image)/255.0

    # As the CNN was trained with batches (extra dimension), we need to add an extra dimension to the image array
    test_image = np.expand_dims(test_image, axis = 0)
    
    # Predict a single result
    result = cnn.predict(test_image)

    # In order to see what animal is the 0 and the 1
    # training_set.class_indices
    if result[0][0] > 0.5:
      prediction = 'dog'
    else:
      prediction = 'cat'
    return (f'The image is a {prediction}')
    
# Function to upload an image
def upload_image():
    filepath = filedialog.askopenfilename()
    global image_path
    image_path = filepath
    if filepath:
        image = Image.open(filepath)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        output_text.set('')

# Function to predict
def predict():
    result = test_image(image_path)
    output_text.set(result)

# Create main window
root = tk.Tk()
root.minsize(800, 600)

# Create widgets
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
image_label = tk.Label(root)
predict_button = tk.Button(root, text="Predict", command=predict)
output_text = tk.StringVar()
output_label = tk.Label(root, textvariable=output_text)

# Layout widgets
upload_button.pack()
image_label.pack()
predict_button.pack()
output_label.pack()

# Start main loop
root.mainloop()


