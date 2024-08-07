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
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO
import zipfile




# Load the dataset CSV file 
ds = pd.read_csv('eda.csv', quoting=3, on_bad_lines='skip')
dataset_file = "questiontrained.csv"
df = pd.read_csv(dataset_file, encoding="latin-1")  # or encoding="ISO-8859-1"
questions_column = "Title"
xtrain = df[questions_column].tolist()

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Prediction", "EDA"],
        default_index=0
    )

#option 1
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
        #function to rpedict similiar question
    def find_similar_questions(input_text, num_similar_questions=5):
     vectorizer = TfidfVectorizer()
     xtrain_tfidf = vectorizer.fit_transform(xtrain)

     input_tfidf = vectorizer.transform([input_text])
     input_tfidf_dense = input_tfidf.toarray()
     xtrain_tfidf_dense = xtrain_tfidf.toarray()

     similarities = cosine_similarity(input_tfidf_dense, xtrain_tfidf_dense).flatten()
     similar_question_indices = similarities.argsort()[-num_similar_questions:][::-1]

     st.write("Top Similar Questions:")
     for i, idx in enumerate(similar_question_indices, start=1):
      st.write(f"{i}. {xtrain[idx]}")

    


    # User input
    st.title("Question Status Prediction")

    title_input = st.text_input("Enter the title:", "")
    bodymark_input = st.text_input("Enter the bodymark:", "")
    tags_input = st.text_input("Enter the tags (space-separated):", "")
#second button
    if st.button("Predict OpenStatus"):
        predictedvaive, predictedmlp = predict_open_status(title_input, bodymark_input, tags_input)
        #st.write("User Input:")
        #st.write("Title:", title_input)
        #st.write("Bodymark:", bodymark_input)
        #st.write("Tags:", tags_input)
        st.write(" Predicted OpenStatus:", "Open" if  predictedvaive == 1 else "Closed")
        if  predictedvaive != 1:  
            open_question_tags = ["python", "data-analysis", "machine-learning", "programming", "help"]
            st.write("Tag Suggestions for closed questions:", open_question_tags)
        
    if st.button("Similar Questions that can be found on StackOverflow:"):
     find_similar_questions(title_input)


#option2:
if selected == "EDA":
    st.title("EDA")
    st.write("A general eda based on the presence of other features in the dataset")
    
    st.subheader("DAY VS POST CREATION DATE")#first analysis

    st.divider()
    ds = pd.read_csv('eda.csv', quoting=3, on_bad_lines='skip')
    ds['PostCreationDate'] = pd.to_datetime(ds['PostCreationDate'], format='%d-%m-%Y %H:%M', errors='coerce')
    day_counts = ds['PostCreationDate'].dt.day_name().value_counts()
    fig,ax=plt.subplots(figsize=(8, 3))
    ax.bar(  day_counts.index, day_counts)
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Number of Posts')
    ax.set_title('Number of Posts for Each Day of the Week')
    st.pyplot(fig)

    st.divider()

    with st.expander("See explanation"):
        st.write("It can be seen that tuesdays have the most number questions being posted per day this may be due to:")
        st.write("Back to the grind: Mondays are often for catching up after the weekend, while Wednesdays involve meetings and planning. Tuesdays leave room for deeper work and technical problem-solving, leading to more questions and answers on Stack Overflow.")
        st.write("Weekly cycle: Tuesdays represent the middle of the work week, when developers might encounter issues they've been struggling with for a couple of days. They have less immediate pressure compared to the beginning or end of the week, allowing them to dedicate time to seeking help or sharing knowledge.")

    st.divider()

#seoncd analysis
    
    st.subheader("MONTH VS POST CREATION DATE")

    ds['PostCreationDate'] = pd.to_datetime(ds['PostCreationDate'], format='%d-%m-%Y %H:%M', errors='coerce')
    month_counts = ds['PostCreationDate'].dt.month_name().value_counts()
    fig,ax=plt.subplots(figsize=(8, 3))
    ax.bar(month_counts.index, month_counts)
    plt.xticks(rotation=45, ha='right')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Posts')
    ax.set_title('Number of Posts for Each Month')
    st.pyplot(fig)
    st.divider()

    with st.expander("See explanation"):
        st.write("the plot shows  that more questions are asked on Stack Overflow during the months of August, September, and October compared to other times of the year.")
        st.write("Begining of fall semester:")
        st.write("August and September mark the beginning of the academic year for many students, particularly those enrolled in computer science or related fields. They may encounter new challenges and concepts, leading them to seek help on Stack Overflow.")
     



     
#third analysis
    st.subheader("YEAR VS POST CREATION DATE")

    ds['PostCreationDate'] = pd.to_datetime(ds['PostCreationDate'], format='%d-%m-%Y %H:%M', errors='coerce')
    year_counts = ds['PostCreationDate'].dt.year.value_counts()
    fig,ax=plt.subplots(figsize=(8, 3))
    ax.bar(year_counts.index, year_counts)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Posts')
    ax.set_title('Number of Posts for Each Year')
    st.pyplot(fig)
    st.divider()

    with st.expander("See explanation"):
        st.write(" It can be seen that there was a spike in posting question ,As the popularity of Stack Overflow gradually increased")

    st.divider()

    #fourth analysis
    st.subheader("THE POPULAR TAGS")

    

    all_tags = ds.loc[:, 'Tag1':'Tag5'].stack().dropna()
    most_popular_tag = all_tags.value_counts().idxmax()
    tag_counts = all_tags.value_counts().head(5)
    st.bar_chart(tag_counts)

    st.divider()

    with st.expander("See explanation"):
        st.write("It can be seen that PHP tags are the most widely used this is due to the fact that:")
        st.write("Legacy Codebases and Maintenance:Many websites and applications built on older PHP versions required maintenance and updates. Stack Overflow became a valuable resource for resolving issues and finding solutions for legacy code.")
        st.write(" Learning and Resource Hub: New developers often chose PHP as an entry point due to its accessibility.Stack Overflow served as a learning platform and knowledge base for beginners seeking guidance.")

    
 
 







