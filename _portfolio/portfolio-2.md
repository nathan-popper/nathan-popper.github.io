---
title: "Customer Service Text Classification"
excerpt: "Best Buy NLP Competition<br><img src='/images/tfidfwordcloud.png' width='500' height='300'>"
collection: portfolio
---

## Project Goal
Train a text classifier to mimic the output of a zero-shot classification from a foundation model. 

### Skills Utilized
- NLP Pipeline Implementations
- Text Vectorization
- Text Feature Extraction
- Handling Imbalanced Classes
- SVM (scikit-learn)

## Data Overview
Best Buy presented us with over 350k custermer interactions that had been transcribed and given class labels with a GPT 3.5 zero-shot model. Overall, the transcriptions were messy and contained a substantial number of typos and formatting issues. Another potential issue was the class imbalance - 5 largest categories accounted for over 42% of the customer interactions

![Label Distribution](/images/label_distribution11.png)

With these challenges in mind, we started working on implementing our NLP Pipeline. 

## NLP Pipeline
Our pipeline included two main steps:
1. Data cleaning
2. Linguistic Preprocessing

  Keeping punctuation and and other irrelevant noise in the text would affect the performance of the SVM model we wanted to use so data cleaning was a valuable first step. We removed punctuation and symbols that had been used to replace redacted information. Every time the speaker changed in the conversation, the transcription would note this by adding "Agent says:" or "Customer says:". I believed that removing these items would cause minimal information loss while reducing the size of the dataset. 
  In our linguistic preprocessing stage I defined function to tokenize, clean out stop-words and lemmatize each conversation. 

{::options parse_block_html="true" /}

<details>
  <summary markdown="span">
    View Preprocessing Code
  </summary>

```python
  from nltk.tokenize import word_tokenize
  from nltk.stem import WordNetLemmatizer
  from nltk.corpus import stopwords

  lemmatizer = WordNetLemmatizer()
  def apply_lemmatizer(text: str) -> str:
      """Apply lemmatizer to a single text conversation"""
      tokens = word_tokenize(text)
      lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
      return " ".join(lemmatized_tokens)

  stop_words = set(stopwords.words('english'))
  def remove_stopwords(text: str) -> str:
      """Apply stop word removal for a single text conversation"""
      tokens = word_tokenize(text)
      token_lst = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
      return " ".join(token_lst)
```
  
</details>

{::options parse_block_html="false" /}

## Vectorization & Training
There were a number of different options for converting the text into numerical vectors that could be fit with a model. Each conversation had largely similar words. Since theses were all calls to Best Buy customer service, the types of phrases and keywords were likely to be similar. In addition, the agents who take the call have very scripted responses which leads conversations to share most of the same words. I didn't want these extremely frequent words to dominate the unique, valuable words. We chose to use TF-IDF vectorization for this reason. 

{::options parse_block_html="true" /}

<details>
  <summary markdown="span">
    View Vectorization Code
  </summary>

```python
  # Set data to train on:  
  X = df["text"]
  y = df["label"]
  
  # Train-Test Split
  X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                      test_size=0.3, 
                                                      random_state=42)
  
  # TF-IDF vectorizing for training X
  vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
  X_train_vectorized = vectorizer.fit_transform(X_train)
```
  
</details>

{::options parse_block_html="false" /}

## Results

## Challenges
- Reduced Vocabulary
- Resampling
- Generative Models

{::options parse_block_html="true" /}

<details>
  <summary markdown="span">
    Code Example
  </summary>

```python
  def func()
```
  
</details>

{::options parse_block_html="false" /}

