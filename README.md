# PRODIGY_DS_04
Analyze and visualize sentiment patterns in social media data

# 💬 Twitter Sentiment Classification & Visualization

This project builds a machine learning model to classify the sentiment of tweets and visualizes sentiment patterns using the **Twitter Entity Sentiment Analysis dataset**.

---

## 📁 Dataset Overview

- **Source**: [Kaggle - Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- **Files Used**:
  - `twitter_training.csv`
  - `twitter_validation.csv`
- **Columns**:
  - `ID`: Unique tweet identifier
  - `Entity`: Mentioned entity in the tweet
  - `Sentiment`: Sentiment label (`Positive`, `Negative`, `Neutral`, `Irrelevant`)
  - `Tweet`: Tweet text
  - (Optional) `tweet_date`: Timestamp of the tweet

---

## 🔄 Workflow

### 1. **Data Loading**
- Loaded training and validation datasets using Pandas.
- Assigned proper column names.

### 2. **Data Cleaning**
- Handled missing values by filling empty tweets with empty strings.
- Removed URLs, mentions, hashtags, special characters using regex.
- Converted all text to lowercase.
- Created a new column `Clean_Tweet` with cleaned text.

### 3. **Label Encoding**
- Encoded sentiment labels into numeric format using `LabelEncoder` for ML modeling.

### 4. **Feature Extraction**
- Applied `TfidfVectorizer` to transform tweets into numerical feature vectors.

### 5. **Model Training**
- Trained a `DecisionTreeClassifier` on the training set.

### 6. **Evaluation**
- Predicted on the validation set.
- Calculated **Top-1 Accuracy** using scikit-learn’s `accuracy_score`.

### 7. **Visualization**
- Created bar charts to show sentiment distribution.
- Generated word clouds for different sentiment categories.
- (Optional) Visualized sentiment trends over time using line plots (if date data available).

---

## 🧪 Results

| Metric          | Score    |
|-----------------|----------|
| Top-1 Accuracy  | XX.XX%   |

*Replace `XX.XX%` with your actual accuracy score.*

---

## 📊 Key Insights

- Cleaning and preprocessing significantly improved text quality for modeling.
- TF-IDF vectorization captured important tweet features.
- Decision Tree provided a simple, interpretable baseline for sentiment classification.
- Visualizations helped in understanding the distribution and common themes in different sentiment classes.

---

## 🧰 Tools & Libraries Used

- Python 3
- Pandas & NumPy — data handling
- scikit-learn — ML modeling and evaluation
- re (regex) — text cleaning
- Matplotlib & Seaborn — visualization
- WordCloud — visualizing common words in tweets

---

## 📌 Future Work

- Experiment with more powerful classifiers like Random Forest, SVM, or BERT-based models.
- Enhance text preprocessing with lemmatization, stopwords removal, emoji handling.
- Deploy a web app for real-time sentiment prediction.
- Analyze sentiment dynamics for specific entities or topics over time.

---

## 🙏 Acknowledgements

- Dataset from [Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- Inspired by real-world applications in marketing, social media monitoring, and public opinion analysis.

