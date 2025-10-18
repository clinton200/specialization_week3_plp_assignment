import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------
# Load and prepare the dataset
# ----------------------------------
@st.cache_data
def load_data():
    iris_data = pd.read_csv("C:/Users/user/Downloads/Iris.csv")
    label_encoder = LabelEncoder()
    iris_data['Species'] = label_encoder.fit_transform(iris_data['Species'])
    return iris_data, label_encoder

iris_data, label_encoder = load_data()
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = iris_data[features]
y = iris_data['Species']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
accuracy = accuracy_score(y_test, clf.predict(X_test))

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.title("🌸 Iris Flower Classification App")
st.markdown("""
This interactive web app uses a **Decision Tree Classifier** to predict the *Iris species*
based on flower measurements.  
You can also explore the dataset and view model performance.
""")

# Sidebar navigation
menu = st.sidebar.radio("📊 Menu", ["Dataset Overview", "Visualizations", "Live Prediction"])

# ----------------------------------
# 1️⃣ Dataset Overview
# ----------------------------------
if menu == "Dataset Overview":
    st.subheader("Dataset Preview")
    st.dataframe(iris_data.head())

    st.subheader("Summary Statistics")
    st.write(iris_data.describe())

    st.subheader("Species Distribution")
    st.bar_chart(iris_data['Species'].value_counts())

    st.success(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# ----------------------------------
# 2️⃣ Visualizations
# ----------------------------------
elif menu == "Visualizations":
    st.subheader("Feature Distributions")

    numerical_columns = iris_data.select_dtypes(include=np.number).columns
    num_plots = len(numerical_columns)

    fig, axes = plt.subplots(1, num_plots, figsize=(3 * num_plots, 4))
    for i, col in enumerate(numerical_columns):
        axes[i].hist(iris_data[col], bins=10, color='orange', edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Pairplot by Species")
    fig2 = sns.pairplot(iris_data, hue='Species', palette='Set2')
    st.pyplot(fig2)

# ----------------------------------
# 3️⃣ Live Prediction
# ----------------------------------
elif menu == "Live Prediction":
    st.subheader("🌼 Enter Flower Measurements")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

    if st.button("🔍 Predict Species"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = clf.predict(input_data)[0]
        predicted_species = label_encoder.inverse_transform([prediction])[0]
        st.success(f"🌸 Predicted Species: **{predicted_species}**")

        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, clf.predict(X_test))
        fig3, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig3)
