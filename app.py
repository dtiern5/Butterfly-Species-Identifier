import PIL
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from classifier import machine_classification, get_percentages, bar_graph_predictions
from connector import create_db_connection, execute_query


def main():
    connection = create_db_connection("us-cdbr-east-04.cleardb.com", "b651608ca7d2bd", "3aa158cc", "heroku_976b1d9f54370b0")
    st.sidebar.header("Dashboard")
    page = st.sidebar.selectbox("Page", ["Image Prediction", "Dataset", "Neural Network"])

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
        uploaded_file = col2.file_uploader("Select a picture: ", type=['png', 'jpeg', 'jpg'])

        if uploaded_file is not None:
            img = PIL.Image.open(uploaded_file)

            with Image.open(uploaded_file) as img:

                (width, height) = (220, 220)
                im_resized = img.resize((width, height))

            col2.image(img, caption='Predicting image...', use_column_width=True)

            img_array = img_to_array(im_resized)
            # print("IMAGE SHAPE:", img_array.shape)
            img_array = img_array / 255
            img_array = np.expand_dims(img_array, axis=0)
            # print("IMAGE EXPANDED SHAPE:", img_array.shape)

            prediction = model.predict(img_array)
            # print("NP.ARGMAX:", np.argmax(prediction))
            label = machine_classification(prediction)
            percentages = get_percentages(prediction)

            bar_graph = bar_graph_predictions(prediction)

            st.write("")
            st.subheader("Prediction:")
            for percentage in percentages[:3]:
                st.write(percentage)

            st.pyplot(fig=bar_graph, clear_figure=None)

            query = f"INSERT INTO heroku_976b1d9f54370b0.dash_log (species) VALUES ('{label}');"
            execute_query(connection, query)

    if page == "Dataset":
        data_example = PIL.Image.open('visualizations/six_random_butterflies.png')
        data_example_code = PIL.Image.open('visualizations/six_random_butterflies_code.png')
        data_distribution = PIL.Image.open('visualizations/images_per_species.png')
        data_distribution_code = PIL.Image.open('visualizations/images_per_species_code.png')


        st.title("Dataset")
        st.text("")

        st.subheader("Example images from dataset")
        st.write("A random selection from the dataset obtained from the following code:")
        st.image(data_example_code)
        st.image(data_example)
        st.text("")
        st.text("")

        st.subheader("Distribution of images for each class in dataset")
        st.write("Visualization of the data distribution plotted with the following code:")
        st.image(data_example_code)
        st.image(data_distribution)



    if page == "Neural Network":
        confusion_matrix = PIL.Image.open('visualizations/matrix_dense.png')
        confusion_matrix_code = PIL.Image.open('visualizations/matrix_dense_code.png')
        clf_report = PIL.Image.open('visualizations/clf_dense.png')
        clf_report_code = PIL.Image.open('visualizations/clf_dense_code.png')
        precision = PIL.Image.open('visualizations/precision.PNG')
        recall = PIL.Image.open('visualizations/recall.PNG')
        f_one = PIL.Image.open('visualizations/f1.PNG')
        pca = PIL.Image.open('visualizations/PCA_with_lines.png')
        pca_code_flatten = PIL.Image.open('visualizations/pca_code_flatten.PNG')
        pca_code_graph = PIL.Image.open('visualizations/pca_code_graph.PNG')
        pair_plot = PIL.Image.open('visualizations/pairplot_kde.png')
        pair_plot_code = PIL.Image.open('visualizations/pairplot_code.PNG')

        st.title("Neural Network Analysis")

        st.subheader("Confusion matrix")
        st.write("Confusion matrix visualization of the DenseNet121 model over the test set")
        st.image(confusion_matrix_code)
        st.image(confusion_matrix)
        st.write("The confusion matrix is showing close to 100% accuracy with classes 1, 4, and 10. "
                 "These correspond to Danaus plexippus, Junonia coenia, and Vanessa cardui respectively. "
                 "Class 5, Lycaena phlaeas is most often confused with Vanessa atalanta and Vanessa cardui. "
                 "Class 3, Heliconius erato has some small errors being identified as Heliconius charitonius.")
        st.text("")
        st.text("")

        st.subheader("Classification report")
        st.image(clf_report_code)
        st.image(clf_report)
        st.image(precision)
        st.image(recall)
        st.image(f_one)
        st.write("While we are achieving 99% accuracy, there are some specific weaknesses in the model around "
                 "Heliconius charitonius and Heliconius erato. The classification report’s recall metric "
                 "is a ratio of true positives to the sum of true positives and false negatives. "
                 "In this case, it means that we are missing true positives of Heliconius charitonius. "
                 "The F1 score is also a touch lower, meaning some of our positive predictions were also "
                 "incorrect.")
        st.text("")
        st.text("")

        st.subheader("Principal component analysis")
        st.image(pca_code_flatten)
        st.image(pca_code_graph)
        st.image(pca)
        st.write("PCA works to reduce the dimensionality of data and compress information by removing redundancies "
                 "in data. Samples and characteristics are not discarded, but instead are flattened into two "
                 "principal components. The eigenvectors have been superimposed onto the image, and X2 and Y2 "
                 "represent the Principal Component axes. This relationship can further be viewed in the "
                 "following Pairplot report.")
        st.text("")
        st.text("")

        st.subheader("Pairplot report")
        st.image(pair_plot_code)
        st.image(pair_plot)


if __name__ == '__main__':
    main()