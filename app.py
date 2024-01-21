import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from keras.models import model_from_json
import numpy as np
from PIL import Image, ImageTk

json_file = open(r"model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights(r"model_weights.h5", by_name=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            image = Image.open(file_path)
            image = image.resize((28, 28))
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.image = photo
            classify_button.config(state=tk.NORMAL)
            global loaded_image
            loaded_image = image.convert('L')
        except Exception as e:
            messagebox.showerror("Error", f"Error occurred: {e}")

def classify_image():
    global loaded_image
    try:
        img_array = np.array(loaded_image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = np.mean(img_array, axis=-1)
            img_array = img_array / 255.0
        elif len(img_array.shape) == 2:
            img_array = img_array / 255.0
        else:
            raise ValueError("Invalid image format")
        img_array = np.asarray(Image.fromarray(img_array).resize((28, 28)))
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=0)
            img_array = np.expand_dims(img_array, axis=-1)
            print(f"Input tensor shape: {img_array.shape}")
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            messagebox.showinfo("Prediction", f"The class is: {class_names[predicted_class]}")
        else:
            raise ValueError("Invalid image shape")
    except Exception as e:
        messagebox.showerror("Error", f"Error occurred: {e}")

def clear_display():
    image_label.config(image=None)
    classify_button.config(state=tk.DISABLED)
    prediction_label.config(text="Prediction: ")
    status_label.config(text="Status: Ready")

root = tk.Tk()
root.title("Fashion MNIST IMAGE CLASSIFICATION")
root.geometry("250x250")
root.configure(bg="lightgray")

image_label = tk.Label(root, bg="white")
image_label.pack(padx=10, pady=10, fill="both")

load_button = tk.Button(root, text="Load Your Image", command=load_image, bg="blue", fg="white")
load_button.pack(pady=5)

classify_button = tk.Button(root, text="Classify", command=classify_image, state=tk.DISABLED, bg="green", fg="white")
classify_button.pack(pady=5)

prediction_label = tk.Label(root, text="Prediction: ")
prediction_label.pack(pady=5)

status_label = tk.Label(root, text="Status: Ready")
status_label.pack(pady=5)

clear_button = tk.Button(root, text="Clear", command=clear_display, bg="red", fg="white")
clear_button.pack(pady=5)

root.mainloop()
 