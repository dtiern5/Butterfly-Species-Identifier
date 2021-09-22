import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
import PIL
from tempfile import NamedTemporaryFile
from classifier import machine_classification, get_percentages, bar_graph_predictions


def main():
    text = "Dashboard"
    page = st.sidebar.selectbox("Dashboard", ["Image Prediction", "Analysis"])

    if page == "Image Prediction":
        st.title("Butterfly Classifier")
        st.header("Using a 10 butterfly dataset to predict the species in an uploaded image")
        st.text("Upload an image to predict a butterfly")

        model = load_model('dense_model.h5')
        uploaded_file = st.file_uploader("Select a picture: ", type="jpg")

        if uploaded_file is not None:
            img = PIL.Image.open(uploaded_file)
            st.image(img, caption='Predicting image...', use_column_width=True)

            img_array = img_to_array(img)
            img_array = cv2.resize(img_array, (220, 220))
            print("image shape: ", img_array.shape)
            img_array = img_array / 255
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)

            label = machine_classification(prediction)
            percentages = get_percentages(prediction)
            bar_graph = bar_graph_predictions(prediction)

            print(label)
            st.write("")
            st.write("Prediction: ", label)
            st.write("")
            for percentage in percentages[:3]:
                st.write(percentage)

            st.pyplot(fig=bar_graph, clear_figure=None)

    if page == "Analysis":
        st.title("Butterfly Classifier Analysis")
        st.header("Cool charts and graphs")
        st.text("Here's some text")

        confusion_matrix = PIL.Image.open('visualizations/dense_confusion_matrix.png')
        data_example = PIL.Image.open('visualizations/six_random_butterflies.png')
        data_distribution = PIL.Image.open('visualizations/images_per_species.png')
        clf_report = PIL.Image.open('visualizations/dense_clf_report.png')
        pca = PIL.Image.open('visualizations/PCA.png')
        pair_plot = PIL.Image.open('visualizations/pairplot_kde.png')

        st.text("Example images from dataset")
        st.image(data_example)
        st.text("")

        st.text("Distribution of images for each class in dataset")
        st.image(data_distribution)
        st.text("")

        st.text("Confusion matrix")
        st.image(confusion_matrix)
        st.text("")

        "Classification report"
        st.image(clf_report)
        st.text("")

        "Principal component analysis"
        st.image(pca)

        "Pairplot report"
        st.image(pair_plot)


if __name__ == '__main__':
    main()