import cv2
import numpy as np
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from skimage.feature import hog

def compute_hog_features(images):
    hog_features = []
    for image in images:
        image_resized = cv2.resize(image, (256, 256))
        features = hog(image_resized, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(4, 4), block_norm='L2-Hys')
        hog_features.append(features)
    return np.array(hog_features)

def load_images_and_faces(directory_path):
    face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
    images = []
    labels = []
    for label in os.listdir(directory_path):
        label_folder = os.path.join(directory_path, label)
        if os.path.isdir(label_folder):
            for image_file in os.listdir(label_folder):
                image_path = os.path.join(label_folder, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    for (x, y, w, h) in faces:
                        face_image = image[y:y+h, x:x+w]
                        images.append(face_image)
                        labels.append(label)
    return images, labels

def gather_features(data_directory):
    X_train = None
    train_folder = os.path.join(data_directory, 'train')
    if os.path.exists(train_folder):
        X_train_faces, y_train_labels = load_images_and_faces(train_folder)
        test_folder = os.path.join(data_directory, 'test')
        X_train_features = compute_hog_features(X_train_faces)
        
        if os.path.exists(test_folder):
            X_test_faces, y_test_labels = load_images_and_faces(test_folder)
            X_test_features = compute_hog_features(X_test_faces)
        else:
            print(f"Warning: 'test' folder not found in {test_folder}")
    else:
        print(f"Warning: 'train' folder not found in {train_folder}")

    return X_train_features, y_train_labels, X_test_features, y_test_labels

def train_classifier(data_paths):

    X_train_feats, y_train, X_test_feats, y_test = gather_features(data_paths[0])

    # First decision tree classifier
    classifier1 = DecisionTreeClassifier(max_depth=6, random_state=42)
    classifier1.fit(X_train_feats, y_train)

    y_pred1 = classifier1.predict(X_test_feats)
    accuracy1 = accuracy_score(y_test, y_pred1)
    conf_matrix1 = confusion_matrix(y_test, y_pred1)
    class_report1 = classification_report(y_test, y_pred1)

    # Second decision tree classifier
    X_train_feats, y_train, X_test_feats, y_test = gather_features(data_paths[1])
    classifier2 = DecisionTreeClassifier(max_depth=8, random_state=23)
    classifier2.fit(X_train_feats, y_train)

    y_pred2 = classifier2.predict(X_test_feats)
    accuracy2 = accuracy_score(y_test, y_pred2)
    conf_matrix2 = confusion_matrix(y_test, y_pred2)
    class_report2 = classification_report(y_test, y_pred2)

    return classifier1, accuracy1, conf_matrix1, class_report1, classifier2, accuracy2, conf_matrix2, class_report2

def compute_single_face_hog(image):
    image_resized = cv2.resize(image, (256, 256))
    features, hog_image = hog(image_resized, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(4, 4), block_norm='L2-Hys', visualize=True)
    return features, hog_image

def predict_face_emotion(image_path, model):

    face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not open image {image_path}")

    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        raise ValueError("No faces found in the image")

    (x, y, w, h) = faces[0]
    face_image = image[y:y+h, x:x+w]

    features, hog_img = compute_single_face_hog(face_image)

    features = features.reshape(1, -1)

    emotion_prediction = model.predict(features)[0]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_with_box = image_rgb.copy()
    cv2.rectangle(image_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image_with_box, emotion_prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image_rgb, image_with_box, hog_img, emotion_prediction

def evaluate_single_image(image_path, model):
    image_rgb, image_with_box, hog_img, emotion = predict_face_emotion(image_path, model)
    return image_rgb, image_with_box, hog_img, emotion
