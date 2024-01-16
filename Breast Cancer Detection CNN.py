# Convolutional Neural Network AI Algorithm Program for Breast Cancer Detection
# Author: Paul Strebenitzer
# For Bachelor Thesis

# Based on the kaggle breast_cancer_CNN notebook by Joshua Ampofo Yentumi, Version 41 of 42 (last successful run)
# Source: https://www.kaggle.com/code/joshuaampofoyentumi/breast-cancer-cnn (first accessed 28.11.2023)

# For now run this program from the "Projekt" folder, not from within the venv folder

# --------------------

# Importing libraries
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

start_time = time.time()

# Loading Meta Data
df_metadata = pd.read_csv('breastcancer_detection_p3.11/Dataset/csv/meta.csv') 
print("-- Metadata DataFrame:\n", df_metadata.head())

# Loading Dicom Info Data
df_dicom = pd.read_csv('breastcancer_detection_p3.11/Dataset/csv/dicom_info.csv')
print("-- Digital Imaging and Communications in Medicine - Info DataFrame:\n", df_dicom.head())
print("-- Different Image Types:\n", df_dicom['SeriesDescription'].unique())
print("-- Image Types Value Count:\n", df_dicom['SeriesDescription'].value_counts())

# Loading Dicom Images
cropped_images = df_dicom[df_dicom['SeriesDescription'] == 'cropped images'].image_path
full_mammogram_images = df_dicom[df_dicom['SeriesDescription'] == 'full mammogram images'].image_path
roi_mask_images = df_dicom[df_dicom['SeriesDescription'] == 'ROI mask images'].image_path

# Set Image Path for Image Types
imagedir = '../breastcancer_detection_p3.11/Dataset/jpeg'
# Change Image Paths for Image Types
cropped_images = cropped_images.replace('CBIS-DDSM/jpeg', imagedir, regex=True)
full_mammogram_images = full_mammogram_images.replace('CBIS-DDSM/jpeg', imagedir, regex=True)
roi_mask_images = roi_mask_images.replace('CBIS-DDSM/jpeg', imagedir, regex=True)

# Print new Image Paths for Image Types
print("-- Cropped Images:\n", cropped_images.iloc[0])
print("-- Full Mammogram Images:\n", full_mammogram_images.iloc[0])
print("-- ROI Mask Images:\n", roi_mask_images.iloc[0])

# Organize Image Paths to key-value pairs 
full_mammogram_images_dict = dict()
cropped_images_dict = dict()
roi_mask_images_dict = dict()

# Split every word between / and and take the long number code as key and path as value
for dicom in full_mammogram_images:
    key = dicom.split('/')[4]
    full_mammogram_images_dict[key] = dicom
for dicom in cropped_images:
    key = dicom.split('/')[4]
    cropped_images_dict[key] = dicom
for dicom in roi_mask_images:
    key = dicom.split('/')[4]
    roi_mask_images_dict[key] = dicom

# Print Organized Image Paths
#print(next(iter(full_mammogram_images_dict.items())))

# --------------------

# MASS DATASET 
print("-- MASS DATASET --")
# Load Mass-Training/Test-Description
df_mass_train = pd.read_csv('breastcancer_detection_p3.11/Dataset/csv/mass_case_description_train_set.csv')
df_mass_test = pd.read_csv('breastcancer_detection_p3.11/Dataset/csv/mass_case_description_test_set.csv')

# Remove all " from the data
df_mass_train = df_mass_train.replace('"','', regex=True)
df_mass_test = df_mass_test.replace('"','', regex=True)

# Change Image Paths (from dicom to jpeg)
def change_path(data):
    for index, image in enumerate(data.values):
        image_name = image[11].split('/')[2]
        data.iloc[index, 11] = full_mammogram_images_dict[image_name]
        image_name = image[12].split('/')[2]
        data.iloc[index, 12] = cropped_images_dict[image_name]

# Apply change_path function to Mass-Training/Test-Description
change_path(df_mass_train)
change_path(df_mass_test)

# Unique Values in pathology column
print("-- Unique Values in pathology column:\n", df_mass_train.pathology.unique())

# Dataframe infos
print("-- Train Dataframe info:\n", df_mass_train.info())
print("-- Test Dataframe info:\n", df_mass_test.info())
# Check missing values
print("-- Missing Train Values:\n", df_mass_train.isnull().sum())
print("-- Missing Test Values:\n", df_mass_test.isnull().sum())

# Fill missing values with backward fill (missing value filled with next value in column)
# train set
df_mass_train['mass shape'] = df_mass_train['mass shape'].bfill()
df_mass_train['mass margins'] = df_mass_train['mass margins'].bfill()
# test set
df_mass_test['mass margins'] = df_mass_test['mass margins'].bfill()


# Check missing values
print("-- Missing Train Values after fillna:\n", df_mass_train.isnull().sum())
print("-- Missing Test Values after fillna:\n", df_mass_test.isnull().sum())

# Quantitative Info
print("-- Quantitative Train Info:\n", df_mass_train.describe())
print("-- Quantitative Test Info:\n", df_mass_test.describe())

#Check shapes
print("-- Shape Train:\n", df_mass_train.shape)
print("-- Shape Test:\n", df_mass_test.shape)

# --------------------

# MASS DATASET VISUALIZATIONS
print("-- MASS DATASET VISUALIZATIONS --")

show_plots = False

if show_plots == True:
    # Pathology Types
    def plot_breast_cancer_types(df):
        value = df['pathology'].value_counts()
        plt.figure(figsize=(10, 5))
        plt.pie(value, labels=value.index, autopct='%1.1f%%')
        plt.title('Breast Cancer Types')
        plt.savefig('breastcancer_detection_p3.11/Plots/Breast_Cancer_Types.png')
        plt.show()

    plot_breast_cancer_types(df_mass_train)

    # Assesment Types
    def plot_assessment_types(df):
        plt.figure(figsize=(10, 5))
        sns.countplot(x='assessment', data=df, hue='pathology')
        plt.title('Assessment Types\n 0: Undetermined, 1: Well Differentiated, 2: Moderately Differentiated, 3: Poorly Differentiated, 4/5: Undifferentiated')
        plt.ylabel('Count')
        plt.xlabel('Assessment')
        plt.savefig('breastcancer_detection_p3.11/Plots/Assessment_Types.png')
        plt.show()

    plot_assessment_types(df_mass_train)

    # Cancer finesse / subtlety
    def plot_cancer_subtlety(df):
        plt.figure(figsize=(10, 5))
        sns.countplot(x='subtlety', data=df)
        plt.title('Cancer Subtlety')
        plt.ylabel('Count')
        plt.xlabel('Subtlety')
        plt.savefig('breastcancer_detection_p3.11/Plots/Cancer_Subtlety.png')
        plt.show()

    plot_cancer_subtlety(df_mass_train)

    # Mass Shape against pathology
    def plot_mass_shape(df):
        plt.figure(figsize=(10, 6))
        sns.countplot(x='mass shape', data=df, hue='pathology')
        plt.title('Mass Shape Types against Pathology')
        plt.ylabel('Pathology Count')
        plt.xlabel('Mass Shape')
        plt.legend()
        plt.xticks(rotation=45, ha='right')
        plt.savefig('breastcancer_detection_p3.11/Plots/Mass_Shape_against_Pathology.png')
        plt.show()

    plot_mass_shape(df_mass_train)

    # Breast Density against pathology
    def plot_breast_density(df):
        plt.figure(figsize=(10, 6))
        sns.countplot(x='breast_density', data=df, hue='pathology')
        plt.title('Breast Density against Pathology\n 1: Fatty, 2: Scattered fibroglandular densities, 3: Heterogeneously dense, 4: Extremely dense')
        plt.ylabel('Pathology Count')
        plt.xlabel('Breast Density')
        plt.legend()
        plt.savefig('breastcancer_detection_p3.11/Plots/Breast_Density_against_Pathology.png')
        plt.show()

    plot_breast_density(df_mass_train)

    # Dissplay Images # error no such file or directory?
    import matplotlib.image as mpimg

    def display_images(column, number):
        number_to_visualize = number
        rows = 1
        cols = number_to_visualize
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
        for index, row in df_mass_train.head(number_to_visualize).iterrows():
            image_path = row[column]
            print(image_path)
            image_path = image_path.replace('../', '')
            image = mpimg.imread(image_path)
            ax = axes[index]
            ax.imshow(image, cmap='gray')
            ax.set_title(f"{row['pathology']}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # Display Images
    print("-- Full Mammograms --")
    display_images('image file path', 4)
    print("-- Cropped Mammograms --")
    display_images('cropped image file path', 4)

# --------------------

input("Press Enter to continue...")

# IMAGE PREPROCESSING
print("-- IMAGE PREPROCESSING --")

# Importing libraries
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Show image before processing
def show_image(image_path):
    image_path = image_path.replace('../', '')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title("Sample Image")
    plt.savefig('breastcancer_detection_p3.11/Plots/Sample_Image_Before.png')
    plt.show()

show_image(df_mass_train['image file path'].iloc[0])

# Image preprocessor
def preprocess_image(image_path, desired_size):
    image_path = image_path.replace('../', '')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB because cv2 reads images in BGR format

    # Crop image to lessen white edges
    image = image[100:-100, 100:-100, :] # crop 100 pixels from each edge of image

    # Reduce noise in image
    #image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15) # reduce noise in image
    
    # Apply adaptive histogram equalization for contrast adjustment
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # clip limit is the threshold for contrast limiting, tileGridSize is the number of grid cells
    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV) # YUV because HE is typically applied to luminance channel
    image_yuv[:,:,0] = clahe.apply(image_yuv[:,:,0]) # apply adaptive histogram equalization
    image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    
    image = cv2.resize(image, (desired_size[1], desired_size[0])) # resize image to desired size to ensure all images have same size
    image = image / 255.0 # normalize image pixel range from [0, 255] to [0, 1]
    return image

# Merge Train and Test Dataframes
mass_full = pd.concat([df_mass_train, df_mass_test], axis=0)

mass_train = df_mass_train
mass_test = df_mass_test

# Checks:

print("-- Unique Value Counts --")
print("--TRAIN:")
print(mass_train['pathology'].value_counts())
# Calculate ratio of target classes
class_counts = mass_train['pathology'].value_counts()
total_samples = class_counts.sum()
class_ratios = class_counts / total_samples

# Print ratio of target classes
print("-- Class Ratios --")
for class_label, ratio in class_ratios.items():
    print(f"{class_label}: {ratio:.2%}")

print("\n--TEST:")
print(mass_test['pathology'].value_counts())
# Calculate ratio of target classes
class_counts = mass_test['pathology'].value_counts()
total_samples = class_counts.sum()
class_ratios = class_counts / total_samples

# Print ratio of target classes
print("-- Class Ratios --")
for class_label, ratio in class_ratios.items():
    print(f"{class_label}: {ratio:.2%}")

#check if there are common values in both train and test sets
print("-- Common Values in Train and Test Sets --")
# Check for common values in train and test sets
common_values = np.intersect1d(mass_train['cropped image file path'], mass_test['cropped image file path'])
if common_values.size > 0:
    print("Common values found in train and test sets:", common_values)
else:
    print("No common values found in train and test sets.")


# Target Size
desired_size = (224, 224, 3) # width, height, channels

#show a preprocessed image
x,y = mass_train['image file path'].iloc[0], mass_train['pathology'].iloc[0]
plt.imshow(preprocess_image(x, desired_size))
plt.title("Sample Preprocessed Image: {}".format(y))
plt.savefig('breastcancer_detection_p3.11/Plots/Sample_Pre_Image.png')
plt.show()


# Apply preprocessing to train/test data
#mass_train['preprocessed_images'] = mass_train['image file path'].apply(lambda x: preprocess_image(x, desired_size))
#mass_test['preprocessed_images'] = mass_test['image file path'].apply(lambda x: preprocess_image(x, desired_size))

# Apply preprocessing to train data
mass_full['preprocessed_images'] = mass_full['image file path'].apply(lambda x: preprocess_image(x, desired_size))
print("-- Preprocessed Images:\n", mass_full['preprocessed_images'].head())

# Mapper for pathology types
mapper = {'BENIGN_WITHOUT_CALLBACK': 0, 'BENIGN': 0, 'MALIGNANT': 1} # 0: Benign & Benign without Callback = No Cancer, 1: Malignant = Cancer

# Convert preprocessed_images column to numpy array for CNN input
X_resized = np.array(mass_full['preprocessed_images'].tolist())

#train_resized = np.array(mass_train['preprocessed_images'].tolist())
#test_resized = np.array(mass_test['preprocessed_images'].tolist())

# Apply mapper to pathology column
mass_full['labels'] = mass_full['pathology'].replace(mapper)

#mass_train['labels'] = mass_train['pathology'].replace(mapper)
#mass_test['labels'] = mass_test['pathology'].replace(mapper)

# Number of classes
num_classes = len(mass_full['labels'].unique())
#num_classes = len(mass_train['labels'].unique())

# print labels
print("-- Labels:\n", mass_full['labels'].unique())
#print("-- Labels:\n", mass_train['labels'].unique())

# Split data into train, test and validation sets (70, 20, 10)
X_train, X_temp, y_train, y_temp = train_test_split(X_resized, mass_full['labels'], test_size=0.3, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Convert integer labels to one-hot labels for for multi-class classification problem (binary vector)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_val = to_categorical(y_val, num_classes)

#y_train = to_categorical(mass_train['labels'], num_classes)
#y_test = to_categorical(mass_test['labels'], num_classes)

# Split data into test and validation sets (80% for test, 20% for validation)
#X_test, X_val, y_test, y_val = train_test_split(test_resized, y_test, test_size=0.5, random_state=42)

# Print the shapes of the new datasets
print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Validation set shape:", X_val.shape)

#print("Train set shape:", train_resized.shape)
#print("Test set shape:", X_test.shape)
#print("Validation set shape:", X_val.shape)

#X_train = train_resized

# --------------------
input("Press Enter to continue...")

# Neural Network Model
print("-- NEURAL NETWORK MODEL --")

# Import Tensorflow Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

def cnn_architecture():
    # Augment Data (artificially increase size of training set by applying random image transformations)
    train_datagen = ImageDataGenerator(rotation_range=0, # rotate image randomly up to 90 degrees (was 40) Or 0 because medical images like X-rays or MRIs are usually taken in a standard orientation
                                    #width_shift_range=0.2, # shift image horizontally randomly up to 20% of image width
                                    #height_shift_range=0.2, # shift image vertically randomly up to 20% of image height
                                    shear_range=0.2, # Shear Transformation up to 20% (distortion along an axis)
                                    zoom_range=0.1, # zoom in randomly up to 10% (was 20)
                                    horizontal_flip=True, # flip image horizontally
                                    fill_mode='constant', # fill in any newly created pixels with nearest filled value # other options: constant, nearest, wrap, reflect
                                    cval = 0 # value used for fill_mode = constant
                                    )

    # Apply Augmentation to Train Data
    train_datagen_augmented = train_datagen.flow(X_train, y_train, batch_size=32) # flow generates batches of randomly transformed images 
    # (how many samples will be propagated through the network at a time; lower batch size = more robust convergence)

    #plot a picture to see how the augmentation looks like
    x,y = train_datagen_augmented.next()
    image = x[0]
    plt.imshow(image)
    plt.title("Augmentation Demo Picture")
    plt.savefig('breastcancer_detection_p3.11/Plots/AugmentDemo.png')
    plt.show()
    # randomly select X images from augmented data and store them in Augment folder
    for i in range(0, 20):
       x,y = train_datagen_augmented.next()
       image = x[0]
       plt.imshow(image)
       plt.title("Augmentation Picture {}".format(i))
       plt.savefig('breastcancer_detection_p3.11/Plots/Augment/Augment{}.png'.format(i))
       print("-- Augmentation Picture {} saved".format(i))

    # Build CNN Model
    model = Sequential() # layers are added one by one
    # Add Convolutional Layers
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3))) 
    # Convolutional Layers perform feature extraction by applying filters to the input and passing the result to the next layer
    # filters: matrices applied to the input to extract features (detecting patterns (edges, lines, shapes, etc.))); number of filters = number of features 
    # kernel_size: dimensions of the filter matrix
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Max Pooling Layers reduce spacial dimensions of input data, helping to extract the most important features while reducing computational complexity
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Dropout Layers randomly drop neurons during training to prevent overfitting
    model.add(Flatten())
    # Flatten Layer converts the output of the previous layer to a 1D array
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal')) # rectified linear unit (max(0, x)) help to mitigate the vanishing gradient problem
    # Dense Layer is a fully connected layer with 512 units (neurons) and ReLU activation function
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal')) # additional dense layer
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal')) # output layer; softmax converts vector of values to probability distribution

    # Compile Model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
    # compile() configures the learning process before training
    # Adam (Adaptive Moment Estimation) calculates adaptive learning rates for each parameter
    # 0.0001 small enopugh to not overshoot the global minimum but large enough to converge quickly

    # Train Model
    history = model.fit(train_datagen_augmented, 
                        epochs=20, # number of times the model will cycle through the data
                        validation_data=(X_val, y_val)
                    )
                    # fit() trains the model 

    # Save Model
    model.save('breastcancer_detection_p3.11/Models/breast_cancer_detection_model.h5')

    return model, history

model, history = cnn_architecture()

print("-- Model Summary:\n")
model.summary()

print("-- Model Evaluation:\n", model.evaluate(X_test, y_test))

# Plot Model
plot_model(model, to_file='breastcancer_detection_p3.11/Plots/Model.png', show_shapes=True, show_layer_names=True)

# --------------------

# Classification Report
print("-- Classification Report --")
from sklearn.metrics import classification_report, confusion_matrix

# confusion matrix labels
cm_labels = ['MALIGNANT', 'BENIGN']

# Predictions
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)
# predict() generates output predictions for the input samples
# Model´s predicted labels for input data

# convert predicted labels to class labels
y_pred_classes_test = np.argmax(y_pred_test, axis=1)
y_pred_classes_train = np.argmax(y_pred_train, axis=1)
# Returns the indices of the maximum values along an axis

y_true_classes_test = np.argmax(y_test, axis=1)
y_true_classes_train = np.argmax(y_train, axis=1)

# Classification Report
test_report = classification_report(y_true_classes_test, y_pred_classes_test, target_names=cm_labels)
print("-- Test Report:\n", test_report)
train_report = classification_report(y_true_classes_train, y_pred_classes_train, target_names=cm_labels)
print("-- Train Report:\n", train_report)

# Confusion Matrix
print("-- Confusion Matrix --")
test_cm = confusion_matrix(y_true_classes_test, y_pred_classes_test)
print("-- Test Confusion Matrix:\n", test_cm)
train_cm = confusion_matrix(y_true_classes_train, y_pred_classes_train)
print("-- Train Confusion Matrix:\n", train_cm)

# Print Confusion Matrix
def plot_confusion_matrix(cm, labels, title):
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_conf_mx = cm / row_sums
    plt.figure(figsize=(10, 10))
    sns.heatmap(norm_conf_mx, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('breastcancer_detection_p3.11/Plots/Confusion_Matrix.png')
    plt.show()

plot_confusion_matrix(test_cm, cm_labels, 'Test Confusion Matrix')
plot_confusion_matrix(train_cm, cm_labels, 'Train Confusion Matrix')

# --------------------

# ROC / AUC Curves

from sklearn.metrics import roc_curve, auc

# Use trained model to predict probabilities
y_pred_prob = model.predict(X_test)

# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)
# fpr: false positive rate, tpr: true positive rate

# Plot ROC Curve
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc, )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig('breastcancer_detection_p3.11/Plots/ROC_Curve.png')
    plt.show()


plot_roc_curve(fpr, tpr, roc_auc)

# Print AUC score
print("AUC Score: {:.2f}".format(roc_auc))

# --------------------

# Plotting Accuracy and Loss
print("-- ACCURACY AND LOSS --")

history_dict = history.history
# history.history is a dictionary containing data about everything that happened during training
print("-- History Dictionary:\n", history_dict.keys())

# Plotting training loss vs validation loss
def plot_loss(history_dict):
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    accuracy = history_dict['accuracy']

    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, loss_values, 'b', label='Training Loss')
    plt.plot(epochs, val_loss_values, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=12)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('breastcancer_detection_p3.11/Plots/LossVal.png')
    plt.show()

print("-- Plot Loss:")
plot_loss(history_dict)

# Plotting training accuracy vs validation accuracy
def plot_accuracy(history_dict):
    accuracy = history_dict['accuracy']
    val_accuracy = history_dict['val_accuracy']

    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=12)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('breastcancer_detection_p3.11/Plots/AccVal.png')
    plt.show()

print("-- Plot Accuracy:")
plot_accuracy(history_dict)

# --------------------

# Demo
print("-- DEMO --")

predictions = model.predict(X_test)

import random

#reverse mapper
reverse_mapper = {v:k for k,v in mapper.items()}

#map predictions to class labels
predictions_class_indices = np.argmax(predictions, axis=1)
predictions_class_names = [reverse_mapper[i] for i in predictions_class_indices]

ground_truth_class_indices = np.argmax(y_test, axis=1)
ground_truth_class_names = [reverse_mapper[i] for i in ground_truth_class_indices]

# display predicted class_names
print("-- Predicted Class Names --")
num_image_visulize = min(10, len(X_test))

# create random indices to select images to display
random_indices = random.sample(range(len(X_test)), num_image_visulize)

# create figure with sub-plots
fig, axes = plt.subplots(2, num_image_visulize // 2, figsize=(15, 10))

for i, index in enumerate(random_indices):
    row = i // (num_image_visulize // 2)
    col = i % (num_image_visulize // 2)
    axes[row, col].imshow(X_test[index])
    axes[row, col].set_title("Prediction: {}\n Ground Truth: {}".format(predictions_class_names[index], ground_truth_class_names[index]))
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig('breastcancer_detection_p3.11/Plots/Demo.png')
plt.show()


# --------------------
print(" \n --- PROGRAM END ---")
print("--- Execution time: {:.2f} minutes ---".format((time.time() - start_time)/60))
