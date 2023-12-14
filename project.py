# Common
import os
import numpy as np
import tensorflow as tf

# Data
import imgaug.augmenters as iaa
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Model
from sklearn.svm import SVC
from keras.models import Sequential
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Optimization
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

# Visualization
import matplotlib.pyplot as plt

# Function to load images from directories
def load_images_from_folder(folder, flatten):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64,64), color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # Flatten image into 1D array
        if flatten:
            images.append(img_array.flatten())
        else:
            images.append(img_array)
        labels.append(folder)
    return np.array(images), labels

# Directory paths for bikes and cars images
bikes_folder = 'Car-Bike-Dataset/Bike'
cars_folder = 'Car-Bike-Dataset/Car'

bike_images, bike_labels = load_images_from_folder(bikes_folder, False)
car_images, car_labels = load_images_from_folder(cars_folder, False)

all_images = np.vstack((bike_images, car_images))
all_labels = np.array(bike_labels + car_labels)

bike_images_flat, bike_labels_flat = load_images_from_folder(bikes_folder, True)
car_images_flat, car_labels_flat = load_images_from_folder(cars_folder, True)

all_images_flat = np.vstack((bike_images_flat, car_images_flat))
all_labels_flat = np.array(bike_labels_flat + car_labels_flat)

print(all_images.shape)
print(all_images_flat.shape)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=44)
X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(all_images_flat, all_labels_flat, test_size=0.2, random_state=44)

# Perform validation split
X_val, X_train = np.split(X_train, [800])
y_val, y_train = np.split(y_train, [800])

X_val_flat, X_train_flat = np.split(X_train_flat, [800])
y_val_flat, y_train_flat = np.split(y_train_flat, [800])

# Apply fit_transform to labels to make them numeric
lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_val = lb.fit_transform(y_val)
y_test = lb.fit_transform(y_test)

print(all_images.shape)
print(all_images_flat.shape)
print()
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

# Function for image augmentation using imgaug
def apply_image_augmentation(images, labels):
    # Define an augmentation pipeline
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        # iaa.Crop(percent=(0, 0.1)),  # random crops
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),  # random Gaussian blur
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))),  # random noise
    ], random_order=True)

    augmented_images = seq(images=images)
    augmented_labels = labels

    return augmented_images, augmented_labels

# Apply image augmentation to training data
X_train_aug, y_train_aug = apply_image_augmentation(X_train, y_train)
X_train_aug = np.vstack((X_train_aug, X_train))
y_train_aug = np.hstack((y_train_aug, y_train))

X_train_aug_flat, y_train_aug_flat = apply_image_augmentation(X_train_flat, y_train)
X_train_aug_flat = np.vstack((X_train_aug_flat, X_train_flat))
y_train_aug_flat = np.hstack((y_train_aug_flat, y_train))

k_values = range(5, 31, 2)
knn_accuracies = []
knn_aug_accuracies = []
for k_value in k_values:
    # Without Augmentation
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train_flat, y_train)
    accuracy = knn.score(X_val_flat, y_val)
    knn_accuracies.append(accuracy)

    # With Augmentation
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train_aug_flat, y_train_aug)
    accuracy = knn.score(X_val_flat, y_val)
    knn_aug_accuracies.append(accuracy)

# Best model for kNN
knn_best_param = k_values[np.argmax(knn_accuracies)]
best_knn = KNeighborsClassifier(n_neighbors=knn_best_param)
best_knn.fit(X_train_flat, y_train)
accuracy = best_knn.score(X_val_flat, y_val)
print(f"Best parameters for kNN: {knn_best_param}")
print(f"Accuracy of the kNN model with parameter optimization: {accuracy:.4f}")

knn_aug_best_param = k_values[np.argmax(knn_aug_accuracies)]
best_knn_aug = KNeighborsClassifier(n_neighbors=knn_aug_best_param)
best_knn_aug.fit(X_train_aug_flat, y_train_aug)
accuracy = best_knn_aug.score(X_val_flat, y_val)
print(f"Best parameters for kNN with augmentation: {knn_aug_best_param}")
print(f"Accuracy of the kNN model with parameter optimization with augmentation: {accuracy:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, knn_aug_accuracies, marker="o", label="Augmented")
plt.plot(k_values, knn_accuracies, marker="o", label="Non-Augmented")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.title("Number of Neighbors vs Accuracy (kNN)")
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()

c_values = [1, 10, 100]
kernel = "poly"
# [1, 10, 100]
# ["linear", "poly", "rbf", "sigmoid"]
svm_accuracies = []
svm_aug_accuracies = []
for c_value in c_values:
    # Without Augmentation
    svm = SVC(kernel=kernel, gamma=c_value)
    svm.fit(X_train_flat, y_train)
    y_pred = svm.predict(X_val_flat)
    accuracy = accuracy_score(y_val, y_pred)
    svm_accuracies.append(accuracy)

    # With Augmentation
    svm_aug = SVC(kernel=kernel, gamma=c_value)
    svm_aug.fit(X_train_aug_flat, y_train_aug)
    y_pred = svm_aug.predict(X_val_flat)
    accuracy_aug = accuracy_score(y_val, y_pred)
    svm_aug_accuracies.append(accuracy_aug)
    
    # Best model for SVM
svm_best_param = c_values[np.argmax(svm_accuracies)]
best_svm = SVC(kernel=kernel, gamma=svm_best_param)
best_svm.fit(X_train_flat, y_train)
y_pred = best_svm.predict(X_val_flat)
accuracy = accuracy_score(y_val, y_pred)
print(f"Best parameters for SVM: {svm_best_param}")
print(f"Accuracy of the SVM model with parameter optimization: {accuracy:.4f}")

# Best model for SVM with augmentation
svm_aug_best_param = c_values[np.argmax(svm_aug_accuracies)]
best_svm_aug = SVC(kernel=kernel, gamma=svm_aug_best_param)
best_svm_aug.fit(X_train_aug_flat, y_train_aug)
y_pred = best_svm_aug.predict(X_val_flat)
accuracy = accuracy_score(y_val, y_pred)
print(f"Best parameters for SVM: {svm_aug_best_param}")
print(f"Accuracy of the SVM model with parameter optimization: {accuracy:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(c_values, svm_aug_accuracies, marker="o", label="Augmented")
plt.plot(c_values, svm_accuracies, marker="o", label="Non-Augmented")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.title("C Values vs Accuracy (SVM)")
plt.xticks(c_values)
plt.grid(True)
plt.legend()
plt.show()

n_estimators = [10, 20, 50, 100]
rf_accuracies = []
rf_aug_accuracies = []
for n_estimator in n_estimators:
    # Without Augmentation
    rf = RandomForestClassifier(random_state=42, n_estimators=n_estimator)
    rf.fit(X_train_flat, y_train)
    y_pred = rf.predict(X_val_flat)
    accuracy = accuracy_score(y_val, y_pred)
    rf_accuracies.append(accuracy)

    # With Augmentation
    rf = RandomForestClassifier(random_state=42, n_estimators=n_estimator)
    rf.fit(X_train_aug_flat, y_train_aug)
    y_pred = rf.predict(X_val_flat)
    accuracy = accuracy_score(y_val, y_pred)
    rf_aug_accuracies.append(accuracy)

# Best model for Random Forest
rf_best_param = n_estimators[np.argmax(rf_accuracies)]
best_rf = RandomForestClassifier(random_state=42, n_estimators=rf_best_param)
best_rf.fit(X_train_flat, y_train)
y_pred = best_rf.predict(X_val_flat)
accuracy = accuracy_score(y_val, y_pred)
print(f"Best parameters for Random Forest: {rf_best_param}")
print(f"Accuracy of the Random Forest model with parameter optimization: {accuracy:.4f}")

rf_aug_best_param = n_estimators[np.argmax(rf_aug_accuracies)]
best_rf_aug = RandomForestClassifier(random_state=42, n_estimators=rf_aug_best_param)
best_rf_aug.fit(X_train_aug_flat, y_train_aug)
y_pred = best_rf_aug.predict(X_val_flat)
accuracy = accuracy_score(y_val, y_pred)
print(f"Best parameters for Random Forest: {rf_aug_best_param}")
print(f"Accuracy of the Random Forest model with parameter optimization: {accuracy:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(n_estimators, rf_aug_accuracies, marker="o", label="Augmented")
plt.plot(n_estimators, rf_accuracies, marker="o", label="Non-Augmented")
plt.xlabel("n_Estimators")
plt.ylabel("Accuracy")
plt.title("n_Estimators vs Accuracy (Random Forest)")
plt.xticks(n_estimators)
plt.grid(True)
plt.legend()
plt.show()

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.summary()

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Apply data augmentation and fit the model
model.fit(X_train_aug, y_train_aug, epochs=5, batch_size=64, validation_data = (X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Accuracy of the CNN model: {accuracy * 100:.2f}%")

# Normal
n_components = [1, 50, 100, 250, 500, 1000, 2400]
X_train_flat_pca = []
X_val_flat_pca = []
for components in n_components:
    pca = PCA(n_components=components)
    pca.fit(X_train_flat)
    X_train_flat_pca.append(pca.transform(X_train_flat))
    X_val_flat_pca.append(pca.transform(X_val_flat))

# Augmentation

for i in range(len(n_components)):
    knn = KNeighborsClassifier(n_neighbors=knn_best_param)
    knn.fit(X_train_flat_pca[i], y_train)
    accuracy = knn.score(X_val_flat_pca[i], y_val)
    print("Accuracy:", accuracy)
    
for i in range(len(n_components)):
    knn_aug = KNeighborsClassifier(n_neighbors=knn_aug_best_param)
    knn_aug.fit(X_train_flat_pca[i], y_train)

    accuracy = knn_aug.score(X_val_flat_pca[i], y_val)
    print("Accuracy:", accuracy)
    
for i in range(len(n_components)):
    svm = SVC(kernel="poly") # TODO
    svm.fit(X_train_flat_pca[i], y_train)
    y_pred = svm.predict(X_val_flat_pca[i])
    accuracy = accuracy_score(y_val, y_pred)
    print(accuracy)
    
for i in range(len(n_components)):
    svm_aug = SVC(kernel="poly") # TODO
    svm_aug.fit(X_train_flat_pca[i], y_train)
    y_pred = svm_aug.predict(X_val_flat_pca[i])
    accuracy = accuracy_score(y_val, y_pred)
    print(accuracy)
    
for i in range(len(n_components)):
    rf = RandomForestClassifier(random_state=42, n_estimators=rf_best_param)
    rf.fit(X_train_flat_pca[i], y_train)
    y_pred = rf.predict(X_val_flat_pca[i])
    accuracy = accuracy_score(y_val, y_pred)
    print(accuracy)
    
for i in range(len(n_components)):
    rf_aug = RandomForestClassifier(random_state=42, n_estimators=rf_aug_best_param)
    rf_aug.fit(X_train_flat_pca[i], y_train)
    y_pred = rf_aug.predict(X_val_flat_pca[i])
    accuracy = accuracy_score(y_val, y_pred)
    print(accuracy)