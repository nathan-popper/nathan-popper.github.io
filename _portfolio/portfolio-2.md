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

[image goes here]

With these challenges in mind, we started working on implementing our NLP Pipeline. 

## NLP Pipeline
Our pipeline included two main steps:
1. Data cleaning
2. Linguistic Preprocessing

  Keeping punctuation and and other irrelevant noise in the text would affect the performance of the SVM model we wanted to use so data cleaning was a valuable first step. We removed punctuation and symbols that had been used to replace redacted information. Every time the speaker changed in the conversation, the transcription would note this by adding "Agent says:" or "Customer says:". I believed that removing these items would cause minimal information loss while reducing the size of the dataset. 
  In our linguistic preprocessing stage each conversation would get tokenized, cleaned of stop-words and lemmatized. 

{::options parse_block_html="true" /}

<details>
  <summary markdown="span">
    Preprocessing Code
  </summary>

```python
  lemmatizer = WordNetLemmatizer()
  def apply_lemmatizer(text: str) -> str:
      """Apply lemmatizer to a single text conversation"""
      tokens = word_tokenize(text)
      lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
      return " ".join(lemmatized_tokens)

  # Edit the base stop word list to keep words that are helpful
  stop_words = set(stopwords.words('english'))
  keep_words = ["have", "not", "below", "few", "down"]
  for word in keep_words:
      stop_words.remove(word)

  def remove_stopwords(text: str) -> str:
      """Apply stop word removal for a single text conversation"""
      tokens = word_tokenize(text)
      token_lst = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
      return " ".join(token_lst)
```
  
</details>

{::options parse_block_html="false" /}

## Text Vectorization

## Model Choice

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

