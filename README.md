
# Natural Language Processing Projects: Sentiment Analysis on Restaurant Reviews

This repository contains two natural language processing (NLP) projects focused on sentiment analysis of restaurant reviews. The projects utilize different machine learning algorithms to classify reviews as either positive or negative.

## Project Overview

Both projects perform sentiment analysis on the "Restaurant_Reviews.tsv" dataset. They implement similar data preprocessing steps but differ in the classification algorithms used.

*   **NLP\_NAIVE\_BAYES.ipynb:** This project uses the Naive Bayes algorithm (specifically, Gaussian Naive Bayes) for sentiment classification.
*   **NLP\_Rand\_forest.ipynb:** This project employs the Random Forest algorithm for sentiment classification.

## Dataset

The dataset used is "Restaurant\_Reviews.tsv", which contains restaurant reviews and their corresponding sentiment labels (positive or negative).  The delimiter is tab-separated (`\t`), and quoting is used to handle special characters within the reviews.

## Dependencies

To run these projects, you'll need the following Python libraries:

*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn (sklearn)
*   nltk

You can install these libraries using pip:

```
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

Also, you need to download the stopwords dataset from `nltk`:

```
import nltk
nltk.download('stopwords')
```

## Setup and Execution

1.  **Clone the repository:**

    ```
    git clone 
    cd 
    ```

2.  **Install the dependencies:**

    ```
    pip install -r requirements.txt
    ```

    *(Note: You may need to create a `requirements.txt` file listing the dependencies if one doesn't exist.)*

3.  **Place the Dataset:**
    Ensure that the `Restaurant_Reviews.tsv` dataset is in the same directory as the Jupyter Notebooks.

4.  **Run the Jupyter Notebooks:**

    You can run the notebooks using Jupyter Lab or Jupyter Notebook:

    ```
    jupyter lab NLP_NAIVE_BAYES.ipynb
    jupyter lab NLP_Rand_forest.ipynb
    ```

    or

    ```
    jupyter notebook NLP_NAIVE_BAYES.ipynb
    jupyter notebook NLP_Rand_forest.ipynb
    ```

    Open each notebook and execute the cells in order.

## Code Explanation

Both notebooks follow a similar structure:

1.  **Import Libraries:** Import necessary libraries such as pandas, numpy, matplotlib, seaborn, nltk, and scikit-learn.
2.  **Load Dataset:** Load the `Restaurant_Reviews.tsv` dataset into a pandas DataFrame.
3.  **Data Preprocessing:**
    *   Clean the reviews using regular expressions (`re.sub`) to remove non-alphabetic characters.
    *   Convert reviews to lowercase.
    *   Tokenize the reviews into individual words.
    *   Apply stemming using the Porter Stemmer to reduce words to their root form.
    *   Remove stop words (common English words like "the", "a", "is") using the `nltk.corpus.stopwords` list.  Note that "not" is specifically removed from the stop words list as it can be important for sentiment.
4.  **Feature Extraction:**
    *   Use `CountVectorizer` from scikit-learn to convert the text data into numerical features.  This creates a bag-of-words representation, where each word is a feature, and the value is the count of that word in the review.
    *   The `max_features` parameter limits the number of features to the top 1500 most frequent words.
5.  **Train-Test Split:**
    *   Split the data into training and testing sets using `train_test_split` from scikit-learn.  The test size is 20% of the data, and `random_state` is set for reproducibility.
6.  **Model Training:**
    *   **NLP\_NAIVE\_BAYES.ipynb:**  Train a `GaussianNB` classifier on the training data.
    *   **NLP\_Rand\_forest.ipynb:** Train a `RandomForestClassifier` with 500 estimators, using 'entropy' as the criterion.
7.  **Model Evaluation:** *(This part is missing from the provided code, but should be included)*
    *   Predict the sentiment on the test set.
    *   Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.  You can use `confusion_matrix` and other metrics from `sklearn.metrics`.

## Model Evaluation (To be added in the notebooks)

To properly evaluate the models, add the following code to each notebook after the model training step:

```
from sklearn.metrics import confusion_matrix, accuracy_score

# Predict on the test set
y_pred = classifier.predict(x_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

This will provide insight into how well the models are performing.

## Results

The results of the sentiment analysis will be printed in the output of the Jupyter Notebooks, including the confusion matrix and accuracy score.

