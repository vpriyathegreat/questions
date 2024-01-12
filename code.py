import streamlit as st
from streamlit_option_menu import option_menu
import re
import nltk
nltk.download("stopwords")
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
#from streamlit_option_menu
#import option_menu
#from training import y

ds = pd.read_csv('eda.csv', quoting=3, on_bad_lines='skip')

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Prediction", "EDA"],
        default_index=0
    )

if selected == "Prediction":
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    clf_nb = pickle.load(open("naive_bayes_model.pickle", "rb"))
    mlp_cv = pickle.load(open("mlp_classifier_model.pickle", "rb"))

    # Function to clean and preprocess user input
    def preprocess_input(title, bodymark, tags):
        title_review = re.sub('[^a-zA-Z]', ' ', title)
        title_review = title_review.lower().split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        title_review = [ps.stem(word) for word in title_review if not word in set(all_stopwords)]
        title_review = ' '.join(title_review)

        body_review = re.sub('[^a-zA-Z]', ' ', bodymark)
        body_review = body_review.lower().split()
        body_review = [ps.stem(word) for word in body_review if not word in set(all_stopwords)]
        body_review = ' '.join(body_review)

        combined_review = title_review + ' ' + body_review + ' ' + tags
        return combined_review

    # Function to predict OpenStatus
    def predict_open_status(title, bodymark, tags):
        input_text = preprocess_input(title, bodymark, tags)
        input_tfidf = vectorizer.transform([input_text]).toarray()

        # Predict using Naive Bayes
        predictedvaive = clf_nb.predict(input_tfidf)

        # Predict using MLP Classifier
        predictedmlp = mlp_cv.predict(input_tfidf)

        return predictedvaive, predictedmlp
    def find_similar_questions(input_text, num_similar_questions=5):
        input_tfidf = vectorizer.transform([input_text]).toarray()
        similarities = cosine_similarity(input_tfidf, xtrain_tfidf).flatten()
        similar_question_indices = similarities.argsort()[-num_similar_questions:][::-1]
        print("Top Similar Questions:")
        for i, idx in enumerate(similar_question_indices, start=1):
           print(f"{i}. {xtrain[idx]}")


    # User input
    st.title("Open Status Prediction")

    title_input = st.text_input("Enter the title:", "")
    bodymark_input = st.text_input("Enter the bodymark:", "")
    tags_input = st.text_input("Enter the tags (space-separated):", "")

    if st.button("Predict OpenStatus"):
        predictedvaive, predictedmlp = predict_open_status(title_input, bodymark_input, tags_input)
        st.write("User Input:")
        st.write("Title:", title_input)
        st.write("Bodymark:", bodymark_input)
        st.write("Tags:", tags_input)
       # st.write("Naive Bayes Predicted OpenStatus:", predictedvaive)
        #print("Naive Bayes Predicted OpenStatus:", "Open" if predicted_nb == 1 else "Closed")
        st.write(" Predicted OpenStatus:", "Open" if  predictedvaive == 1 else "Closed")
        #st.write(" Predicted OpenStatus:", predictedmlp)
        if  predictedvaive != 1:  # If not predicted to be an open question
            open_question_tags = ["python", "data-analysis", "machine-learning", "programming", "help"]
            st.write("Suggestions for closed questions:", open_question_tags)
        # else:
        #     st.write("No suggestions for open questions.")
    if st.button("Find Similar Questions"):
            user_input = preprocess_input(title_input, bodymark_input, tags_input)
            find_similar_questions(user_input)



if selected == "EDA":
    st.subheader("DAY VS POST CREATION DATE")

    st.divider()

    ds = pd.read_csv('eda.csv', quoting=3, on_bad_lines='skip')

    # Convert 'PostCreationDate' to datetime
    ds['PostCreationDate'] = pd.to_datetime(ds['PostCreationDate'], format='%d-%m-%Y %H:%M', errors='coerce')

    # Extract day of the week and count occurrences
    day_counts = ds['PostCreationDate'].dt.day_name().value_counts()

    
    fig,ax=plt.subplots(figsize=(8, 3))
    ax.bar(  day_counts.index, day_counts)

# Set plot labels and title
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Number of Posts')
    ax.set_title('Number of Posts for Each Day of the Week')

# Display the plot using Streamlit
    st.pyplot(fig)

    st.divider()

    with st.expander("See explanation"):
        st.write("Back to the grind: Mondays are often for catching up after the weekend, while Wednesdays involve meetings and planning. Tuesdays leave room for deeper work and technical problem-solving, leading to more questions and answers on Stack Overflow.")
        st.write("Weekly cycle: Tuesdays represent the middle of the work week, when developers might encounter issues they've been struggling with for a couple of days. They have less immediate pressure compared to the beginning or end of the week, allowing them to dedicate time to seeking help or sharing knowledge.")

    st.divider()

    st.subheader("MONTH VS POST CREATION DATE")

    ds['PostCreationDate'] = pd.to_datetime(ds['PostCreationDate'], format='%d-%m-%Y %H:%M', errors='coerce')

# Extract month and count occurrences
    month_counts = ds['PostCreationDate'].dt.month_name().value_counts()

    fig,ax=plt.subplots(figsize=(8, 3))
    ax.bar(month_counts.index, month_counts)

# Set plot labels and title
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Posts')
    ax.set_title('Number of Posts for Each Month')
# Display the plot using Streamlit
    st.pyplot(fig)

    st.divider()

    with st.expander("See explanation"):
        st.write()
    


    st.divider()

    st.subheader("YEAR VS POST CREATION DATE")

    ds['PostCreationDate'] = pd.to_datetime(ds['PostCreationDate'], format='%d-%m-%Y %H:%M', errors='coerce')

# Extract month and count occurrences
    year_counts = ds['PostCreationDate'].dt.year.value_counts()

    fig,ax=plt.subplots(figsize=(8, 3))
    ax.bar(year_counts.index, year_counts)

# Set plot labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Posts')
    ax.set_title('Number of Posts for Each Year')
# Display the plot using Streamlit
    st.pyplot(fig)

    st.divider()

    with st.expander("See explanation"):
        st.write()

    st.divider()

    st.subheader("THE POPULAR TAGS")

    

    all_tags = ds.loc[:, 'Tag1':'Tag5'].stack().dropna()

# Find the most popular tag
    most_popular_tag = all_tags.value_counts().idxmax()
    tag_counts = all_tags.value_counts().head(5)

# Plot the bar chart using Streamlit
    st.bar_chart(tag_counts)

    st.divider()

    with st.expander("See explanation"):
        st.write()






