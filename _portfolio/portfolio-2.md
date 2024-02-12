---
title: "Text Classification of Customer Service Calls (Best Buy NLP Competition)"
excerpt: "<img src='/images/employment_def_wordcloud.png' width='500' height='300'>"
collection: portfolio
---

## Project Goal
Train a text classifier to identify the topic of customer service calls made to Best Buy.

- project goal
- strategy
- results

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

<br>
The following wordclouds demonstrate the effectiveness of TF-IDF vectorization for text with frequently repeating words. The size of the word represents term frequency within the "employment or career inquiries" label. 

![Default Employment Wordcloud](/images/employment_def_wordcloud.png)

After adding inverse document frequency, the most unique words within this label become much more dominant. While it's not easy to pick up which label the first wordcloud comes from, there is no doubt that the second word cloud was generated from conversations specifically about getting a new job.

![TFIDF Employment Wordcloud](/images/employment_tfidf_wordcloud.png)


## Results
We began by testing the performance between logistic regression, regressioin trees, and linear support vector machines. The performance in SVM was clearly the best so we decided to move forward with that model. Testing the SVM on the uncleaned text, the model was able to achieve an accuracy of around 55%. After implementing our pipeline and using optimal parameters, this accuracy improved to 62%. For comparison, the Best Buy team trained T5-small model for 20 epochs & achieved an F1 of 72%. While their model had superior performance, the difference in model complexity should be considered. One of the main goals of our project was to balance complexity and performance which is why we chose SVM rather than transformers or another neural network architecture. 

## Challenges
One of the most challenging components of the project was deciding on how to deal with the class imbalance. When using TF-IDF and in classification in general, imbalanced classes tend to bias the results of the model. The following visualization shows the performance of each label in relation to that labels total size. 

![Imbalance-Performance](/images/performance_imbalance12.png)

The groups with the lowest number of observations did perform the worst overall. We attempted to oversample to minority group, undersample the majority group and even different mixes of both. Unfortunately, our changes improved certain groups while hurting others and the overall performance remained the same. We did also experiment with generative models that would consider the prior distributions of the labels but these performed much worse too. 

### An Interesting Finding about Performance
The most interesting discovery came after I spent some time analyzing the vocabulary uniqueness of each label. Using Bayes' theorem I calculated the probability of each label given each word. After running this code I had scores for each word-label combination and could look for the highest value words - words that had high conditional probability. For example, the best word in the entire dataset was "hiring". Over 50% of the time it appeared in the "employment or career inquiries" label while appearing in the entire dataset less than 1% of the time. When the word "hiring" appears there was almost an 80% chance that the conversation could be labeled correctly, and this was without having to account for any other words. I grouped together each label with it's most valuable unique words. Below is a visualization of each label along with a score I calculated to reflect the uniqueness of the label vocabulary.

![label_unique_score](/images/vocab_performance12.png)

Overall, the categories that performed the best were just the ones that had the most unique words to identify the conversations with. If the conversations don't have any words that set it apart from the others it will be hard to classify with even the most complex models. 

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

