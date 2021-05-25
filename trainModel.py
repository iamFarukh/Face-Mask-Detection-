# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# python -m venv --system-site-packages .\venv
# .\venv\Scripts\activate

# Initial Learning Rate
INIT_LR = 1e-4
# No of Epochs
EPOCHS = 20

# Batch size
BS = 32

# Provide the path where training images are Available
DIRECTORY = r"C:\Users\Mohammad Hussain\Desktop\Study Material\Project\Face Mask Detection\dataset"
					
CATEGORIES = ["with_mask", "without_mask"]

# Getting list of images to our directory
print("INFORMATION :- Data is Loading....")

data = []           # In this, we'll append the all the Array of images
labels = []         # This contain that the image is "with_mask" or "without_mask"


# For Data Preprosseing
for category in CATEGORIES:
	# This will get the path of image folder
    path = os.path.join(DIRECTORY, category)
	# os.listdir = list down all the images in side the perticular directory
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# This will converts our lables ("with_mask" or "without_mask") to
# Categorical Values like 0 and 1
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Here we converted these numerical value to array usign Numpy- Array
data = np.array(data, dtype="float32")
labels = np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(
    # the data and their labels
    data, labels,
    # Here we our test data size is 20% and rest 80 % for training
    test_size=0.20,
    stratify=labels,
    random_state=42)
	

# This will create new images by shifting, croping, zooming, roatating etc
# All the possible operation on image
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Loading the MobileNetV2
baseModel = (
    # imagenet is prebuilt in MobileNetV2 specially for images for better result
    weights="imagenet",
    include_top=False,
    # Shape of the image (224x224) and 3 stands for RGB
    input_tensor=Input(shape=(224, 224, 3)))

# Creating the HeadModel
# Passing BaseModel to HeadModel
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
# Here we use A Layer with 128 Nurones
headModel = Dense(128, activation="relu")(headModel)
# Dropout for Overfitting of our model
headModel = Dropout(0.5)(headModel)
# Here a dense Layer with 2 Nurones ("with_mask" or "without_mask")
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# not be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# Compiling the Model
print("INFORMATION :- Compling the Model....")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Training the head
print("INFORMATION :- Training the Head....")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Prediction On testing Data
print("INFORMATION :- Pridiction is Going on....")
predIdxs = model.predict(testX, batch_size=BS)

# Finding the Index of largest Predectible Probabilty
predIdxs = np.argmax(predIdxs, axis=1)

# Now Displaying the Classification Report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# Save the Model
print("INFORMATION :- Saving the Mask Detector Model ....")
model.save("mask_detector.model", save_format="h5")

# In the End
# Plot the Training loss and  accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("EndResut.png")
