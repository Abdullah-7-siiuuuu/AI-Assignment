import models.model as model
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Dark mode style
BACKGROUND_COLOR = "#2E2E2E"
FOREGROUND_COLOR = "#FFFFFF"
BUTTON_COLOR = "#1F1F1F"
HOVER_COLOR = "#3A3A3A"
TEXT_COLOR = "#1E90FF"  # Blue

def select_folder():
    folder_path = filedialog.askdirectory(title="Select a Folder")
    return folder_path

def select_image():
    image_path = filedialog.askopenfilename(title="Select an Image",
                                            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff")])
    return image_path

def output_test_result(img_rgb, img_with_box, hog_image, prediction):
    plt.figure(figsize=(18, 6))

    # subplot 1
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title(f"Original Input Image")
    plt.axis('off')

    # subplot 2
    plt.subplot(1, 3, 2)
    plt.imshow(img_with_box)
    plt.title(f"Predicted Expression {prediction} with detection")
    plt.axis('off')

    # subplot 3
    plt.subplot(1, 3, 3)
    plt.imshow(hog_image, cmap='gray')
    plt.title("HOG features")
    plt.axis('off')

    print(f"Predicted emotion is {prediction}")
    plt.show()

def show_ui():
    # Create window
    window = tk.Tk()
    window.title("FACIAL RECOGNITION")
    window.geometry('700x500')
    window.configure(bg=BACKGROUND_COLOR)

    # Create a frame for layout
    main_frame = tk.Frame(window, bg=BACKGROUND_COLOR)
    main_frame.pack(padx=20, pady=20, fill="both", expand=True)

    # Title Label
    title_label = tk.Label(main_frame, text="FACIAL RECOGNITION", font=("Helvetica", 20), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    title_label.grid(row=0, column=0, columnspan=2, pady=10)

    # Dataset Path Selection
    dataset1_path_var = tk.StringVar(value="No folder selected")
    dataset2_path_var = tk.StringVar(value="No folder selected")

    def select_dataset_folder1():
        folder = select_folder()
        if folder:
            dataset1_path_var.set(folder)

    def select_dataset_folder2():
        folder = select_folder()
        if folder:
            dataset2_path_var.set(folder)

    # Dataset 1 UI components
    dataset1_label = tk.Label(main_frame, text="Select the first folder for the dataset", font=("Helvetica", 12), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    dataset1_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

    dataset1_button = tk.Button(main_frame, text="Select Folder", command=select_dataset_folder1, bg=BUTTON_COLOR, fg=TEXT_COLOR, relief="flat")
    dataset1_button.grid(row=1, column=1, padx=10, pady=10)

    accuracy_label1 = tk.Label(main_frame, text="Accuracy: Not trained yet", font=("Helvetica", 12), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    accuracy_label1.grid(row=2, column=1, padx=10, pady=5, sticky="w")

    # Dataset 2 UI components
    dataset2_label = tk.Label(main_frame, text="Select the second folder for the dataset", font=("Helvetica", 12), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    dataset2_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")

    dataset2_button = tk.Button(main_frame, text="Select Folder", command=select_dataset_folder2, bg=BUTTON_COLOR, fg=TEXT_COLOR, relief="flat")
    dataset2_button.grid(row=3, column=1, padx=10, pady=10)

    accuracy_label2 = tk.Label(main_frame, text="Accuracy: Not trained yet", font=("Helvetica", 12), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    accuracy_label2.grid(row=4, column=1, padx=10, pady=5, sticky="w")

    # Train the model
    def train_model():
        dataset1_path = dataset1_path_var.get()
        dataset2_path = dataset2_path_var.get()

        # Call the new train_classifier function from model.py
        classifier1, accuracy1, conf_matrix1, class_report1, classifier2, accuracy2, conf_matrix2, class_report2 = model.train_classifier([dataset1_path, dataset2_path])

        # Update accuracy labels
        accuracy_label1.config(text=f"Accuracy: {accuracy1 * 100:.2f}%")
        accuracy_label2.config(text=f"Accuracy: {accuracy2 * 100:.2f}%")

        print("Model has been trained")
        print(f"Accuracy 1: {accuracy1 * 100:.2f}%")
        print(f"Confusion Matrix 1:\n{conf_matrix1}")

        print(f"Accuracy 2: {accuracy2 * 100:.2f}%")
        print(f"Confusion Matrix 2:\n{conf_matrix2}")

        global classifier1_global
        classifier1_global = classifier1
        global accuracy1_global
        accuracy1_global = accuracy1
        global confusion_matrix1_global
        confusion_matrix1_global = conf_matrix1
        global classifier2_global
        classifier2_global = classifier2
        global accuracy2_global
        accuracy2_global = accuracy2
        global confusion_matrix2_global
        confusion_matrix2_global = conf_matrix2

    # Train model button
    train_button = tk.Button(main_frame, text="Train Model", command=train_model, bg=BUTTON_COLOR, fg=TEXT_COLOR, relief="flat")
    train_button.grid(row=5, column=0, columnspan=2, pady=20)

    # Test model with single image
    def test_model_single():
        image_path = select_image()
        print(f"Image Path: {image_path}")
        
        # Use the global classifiers to test with the trained models
        img_rgb1, img_with_box1, hog_image1, prediction1 = model.evaluate_single_image(image_path, classifier1_global)
        img_rgb2, img_with_box2, hog_image2, prediction2 = model.evaluate_single_image(image_path, classifier2_global)

        # Output results for both classifiers
        output_test_result(img_rgb1, img_with_box1, hog_image1, prediction1)
        output_test_result(img_rgb2, img_with_box2, hog_image2, prediction2)

    # Test model button (larger size)
    test_button = tk.Button(main_frame, text="Test Model with Image", command=test_model_single, bg=BUTTON_COLOR, fg=TEXT_COLOR, relief="flat", font=("Helvetica", 14), height=2, width=20)
    test_button.grid(row=6, column=0, columnspan=2, pady=20)

    window.mainloop()

# Show the UI
show_ui()
