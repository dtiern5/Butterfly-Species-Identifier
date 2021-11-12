The app can be viewed at https://butterfly-recog.herokuapp.com/

This is me getting my feet wet with TensorFlow and Python's machine learning libraries.
Three models were trained in the .ipnyb Jupyter Notebook file. Accuracy and Loss graphs,
classification reports, and confusion matrices were produced for each.

My custom model got up to an 86% accuracy on the test set, but the DenseNet121 model hit 98%.
With that said, it may be slightly overtrained and has worse performance with identifying Lycaena phlaeas 
and Heliconius charitonius.

Since I was concentrating mostly on the machine learning aspects of this project, I simply used Streamlit and
hosted it on Heroku. I did, however, use this as an opportunity to learn about connecting databases in Python, and 
am storing every instance of a photo upload in a MySql database.
