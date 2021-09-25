import PIL
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from classifier import machine_classification, get_percentages, bar_graph_predictions


def main():
    st.sidebar.header("Dashboard")
    page = st.sidebar.selectbox("Page", ["Image Prediction", "Analysis", "Jupyter Notebook"])

    # Image Prediction page
    if page == "Image Prediction":

        col1, col2 = st.columns(2)
        col1.title("Butterfly Classifier")
        col1.subheader("Accepts the following species:")
        col1.text('1. Danaus plexippus')
        col1.text('2. Heliconius erato')
        col1.text('3. Junonia coenia')
        col1.text('4. Lycaena phlaeas')
        col1.text('5. Nymphalis antiopa')
        col1.text('6. Papilio cresphontes')
        col1.text('7. Pieris rapae')
        col1.text('8. Heliconius charitonius')
        col1.text('9. Vanessa atalanta')
        col1.text('10. Vanessa cardui')



        col2.header("Upload")

        model = load_model('dense_model.h5')
        uploaded_file = col2.file_uploader("Select a picture: ", type="jpg")

        if uploaded_file is not None:
            img = PIL.Image.open(uploaded_file)

            with Image.open(uploaded_file) as img:

                (width, height) = (220, 220)
                im_resized = img.resize((width, height))

            col2.image(img, caption='Predicting image...', use_column_width=True)

            img_array = img_to_array(im_resized)
            print("image shape: ", img_array.shape)
            img_array = img_array / 255
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)

            label = machine_classification(prediction)
            percentages = get_percentages(prediction)
            bar_graph = bar_graph_predictions(prediction)

            st.write("")
            st.subheader("Prediction:")
            for percentage in percentages[:3]:
                st.write(percentage)

            st.pyplot(fig=bar_graph, clear_figure=None)

    if page == "Analysis":
        st.title("Butterfly Classifier Analysis")

        col1, col2, col3 = st.columns(3)

        confusion_matrix = PIL.Image.open('visualizations/dense_confusion_matrix.png')
        data_example = PIL.Image.open('visualizations/six_random_butterflies.png')
        data_distribution = PIL.Image.open('visualizations/images_per_species.png')
        clf_report = PIL.Image.open('visualizations/dense_clf_report.png')
        pca = PIL.Image.open('visualizations/PCA.png')
        pair_plot = PIL.Image.open('visualizations/pairplot_kde.png')
        kmeans = PIL.Image.open('visualizations/K-Means.png')

        col1.text("Example images from dataset")
        col1.image(data_example)
        col1.text("")

        col1.text("Distribution of images for each class in dataset")
        col1.image(data_distribution)
        col1.text("")

        col2.text("Confusion matrix")
        col2.image(confusion_matrix)
        col2.text("")

        col2.text("Classification report")
        col2.image(clf_report)
        col2.text("")

        col3.text("Principal component analysis")
        col3.image(pca)

        col3.text("Pairplot report")
        col3.image(pair_plot)

        col3.text("K-Means")
        col3.image(kmeans)


if __name__ == '__main__':
    main()