# Spam Email Detector

A machine learning project that classifies emails as spam or legitimate using Natural Language Processing (NLP) techniques and classification algorithms.

## Overview

This Spam Email Detector uses machine learning to distinguish between unwanted spam emails and legitimate ones. The system preprocesses email text data, extracts relevant features, and applies either a Naive Bayes or Support Vector Machine (SVM) classifier to make predictions.

## Features

- Text preprocessing including tokenization, stopword removal, and stemming
- Choice of classification algorithms: Naive Bayes or SVM
- Performance evaluation with accuracy, confusion matrix, and detailed classification metrics
- Simple API for making predictions on new emails
- Model persistence (save/load functionality)

## Project Structure

```
spam-email-detector/
│
├── spam_detector.py       # Main implementation of the SpamDetector class
├── example.py             # Example script showing how to use the detector
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── data/                  # Dataset storage
│   └── spam_ham_dataset.csv  # The dataset file
│
└── models/                # Saved models
    └── spam_detector.pkl  # Saved model file
```

## Requirements

- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - nltk

## Installation

1. Clone this repository or download the files:
   ```
   git clone https://github.com/mdzubayerhossain/spam-email-detector.git
   cd spam-email-detector
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download required NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## Getting Started

### Running the Example Script

For a quick demonstration of how the spam detector works:

```
python example.py
```

This will:
1. Create a sample dataset if one doesn't exist
2. Train a Naive Bayes model on the dataset
3. Evaluate the model's performance
4. Test it on new example emails
5. Save and load the model to demonstrate persistence

### Using Your Own Dataset

To train the model with your own dataset:
- Format it as a CSV with at least two columns:
  - `text`: The content of the email
  - `label`: The classification ('spam' or 'ham')
- Place it in the `data/` directory as `spam_ham_dataset.csv`

You can obtain suitable datasets from:
- [Kaggle's Spam Email Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

### Troubleshooting Common Issues

#### Encoding Issues

If you encounter encoding errors with your dataset:

```python
# Modify the data loading line in your code
df = pd.read_csv(dataset_path, encoding='latin1')  # or try 'utf-8', 'cp1252', etc.
```

#### Missing NLTK Resources

If you get errors about missing NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Using the SpamDetector Class

### Training a Model

```python
from spam_detector import SpamDetector

# Create a detector with Naive Bayes
detector = SpamDetector(model_type='naive_bayes')

# Load your dataset
# X_train: list of email texts
# y_train: list of labels (1 for spam, 0 for not spam)

# Train the model
detector.train(X_train, y_train)

# Save the model for later use
detector.save_model('models/spam_detector_nb.pkl')
```

### Making Predictions

```python
# Load a saved model
from spam_detector import SpamDetector
detector = SpamDetector.load_model('models/spam_detector.pkl')

# Classify a single email
email = "Congratulations! You've won a free iPhone. Click here to claim now!"
prediction = detector.predict([email])[0]
probability = detector.predict_proba([email])[0]

print(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
print(f"Probability of being spam: {probability:.4f}")
```

### Evaluating Model Performance

```python
# Evaluate on test data
results = detector.evaluate(X_test, y_test)

print(f"Accuracy: {results['accuracy']:.4f}")
print("Classification Report:")
print(results['classification_report'])
print("Confusion Matrix:")
print(results['confusion_matrix'])
```

## Advanced Usage

### Custom Preprocessing

You can extend the `preprocess_text` method to implement custom preprocessing steps:

```python
def custom_preprocess(text):
    # Basic preprocessing (lowercase, remove special chars)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Your custom steps here
    # ...
    
    return text

# Then use it with your detector
detector.preprocess_text = custom_preprocess
```

### Using Different Models

The project supports both Naive Bayes and SVM models:

```python
# Create a detector with SVM
svm_detector = SpamDetector(model_type='svm')
svm_detector.train(X_train, y_train)

# Compare performance
nb_results = nb_detector.evaluate(X_test, y_test)
svm_results = svm_detector.evaluate(X_test, y_test)

print(f"Naive Bayes accuracy: {nb_results['accuracy']:.4f}")
print(f"SVM accuracy: {svm_results['accuracy']:.4f}")
```

## Performance

The performance of the model depends on the dataset used for training. Typically:

- **Naive Bayes** achieves 95-98% accuracy and is faster to train
- **SVM** can achieve slightly higher accuracy (97-99%) but takes longer to train

## Future Improvements

- Add more classification algorithms (Random Forest, Gradient Boosting)
- Implement cross-validation for better model selection
- Create a simple web interface for interactive classification
- Add email header analysis for better detection
- Support for other languages beyond English

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The NLTK team for providing essential NLP tools
- Scikit-learn for their machine learning implementations
- UCI Machine Learning Repository for dataset resources
