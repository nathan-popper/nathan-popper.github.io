---
title: "Customer Service Text Classification"
excerpt: "Best Buy NLP Competition<br><img src='/images/employment_def_wordcloud.png' width='500' height='300'>"
collection: portfolio
---

## Project Goal
Train a text classifier to identify the topic of customer service calls made to Best Buy

### Skills Utilized
- NLP Pipeline Implementations
- Text Vectorization
- Text Feature Extraction
- Handling Imbalanced Classes
- SVM (scikit-learn)

## Data Overview
Best Buy presented us with over 350k custermer phone calls that had been transcribed and then given labels with a GPT 3.5 zero-shot model. Right from the start, we noticed some unique aspects of this dataset. The transcriptions were messy and contained a substantial number of typos and formatting issues. Even in different call topics, the words and phrases used in most of the conversations was noticeably similar. These were all calls to Best Buy customer service, so the similarity of the types of phrases and keywords is not surprising. Within each call, there were some very common words. For example, every time the speaker changed in the conversation the transcription would note this by adding "Agent says" or "Customer says". Also, the agents who take the call have very scripted responses which leads to additional similarity between conversations. Another potential issue was the class imbalance - the five largest categories accounted for over 42% of the customer interactions. Each of the 57 labels are charted below with their total count. 

![Label Distribution](/images/label_distribution11.png)

## NLP Pipeline
Our pipeline included two main steps:
1. Data cleaning
2. Linguistic Preprocessing

  Keeping punctuation and and other irrelevant noise in the text would affect the performance of the classification model we wanted to use, so data cleaning was a valuable first step. We removed punctuation and symbols that had been used to replace redacted information. In our linguistic preprocessing stage I defined function to tokenize, clean out stop-words and lemmatize each conversation. We then had clean text ready to be used in the classification model. 

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
The classification models used can only work with numbers, and there were a number of different options for converting the cleaned text into the appropriate numerical vectors. Based on the issues mentioned above, TF-IDF vectorization was a good choice. TF-IDF would look at how often a word appears in a certain conversation, and then weight it by how unique that word was to the entire dataset of conversations. Words that appeared in many conversation topics would be less dominant than words that were unique to the smaller subset of labels. 
  I used the scikit-learn TF-IDF vectorizer and customized the parameters to fit well with our situation. I used sublinear term frequency scaling to tone down the effect of very frequent words (some words had 20+ appearances per conversation). A word could also be too unique. If a word only appears in one document it would get quite a large weight because it's correctly identified as a unique word. However, this word is likely not important to identifying the label since it only appeared once. I required that the term appear in at least 3 conversations for it to get added to the vocabulary. Finally, since words alone could miss valuable information, the model was also given bigrams. 

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


The following wordclouds demonstrate the effectiveness of TF-IDF vectorization for text with frequently repeating words. The size of the word represents term frequency within the "employment or career inquiries" label. 

![Default Employment Wordcloud](/images/employment_def_wordcloud.png)

After adding inverse document frequency, the most unique words within this label become much more dominant. While it's not easy to pick up which label the first wordcloud comes from, there is no doubt that the second word cloud was generated from conversations specifically about getting a new job.

![TFIDF Employment Wordcloud](/images/employment_tfidf_wordcloud.png)


## Results

## Challenges

![Imbalance-Performance](/images/performance_imbalance.png)

![label_unique_score](/images/label_unique_f1_2.png)

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

