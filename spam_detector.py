# Spam Email Detector
# A machine learning model to classify emails as spam or legitimate

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class SpamDetector:
    def __init__(self, model_type='naive_bayes'):
        """
        Initialize the spam detector
        
        Parameters:
        -----------
        model_type : str
            The type of model to use, either 'naive_bayes' or 'svm'
        """
        self.model_type = model_type
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        
    def preprocess_text(self, text):
        """
        Preprocess the text by removing special characters, numbers, 
        converting to lowercase, removing stopwords, and stemming
        
        Parameters:
        -----------
        text : str
            The text to preprocess
            
        Returns:
        --------
        str
            The preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and apply stemming
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Join tokens back into a string
        return ' '.join(tokens)
    
    def build_model(self):
        """
        Build the machine learning pipeline
        """
        if self.model_type == 'naive_bayes':
            self.model = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB())
            ])
        elif self.model_type == 'svm':
            self.model = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SVC(kernel='linear', probability=True))
            ])
        else:
            raise ValueError("Model type must be either 'naive_bayes' or 'svm'")
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Parameters:
        -----------
        X_train : array-like
            The training data
        y_train : array-like
            The training labels
        """
        if self.model is None:
            self.build_model()
        
        # Clean the data
        X_train_clean = [self.preprocess_text(text) for text in X_train]
        
        # Train the model
        self.model.fit(X_train_clean, y_train)
        
    def predict(self, X):
        """
        Predict whether an email is spam or not
        
        Parameters:
        -----------
        X : str or list
            The email(s) to classify
            
        Returns:
        --------
        array-like
            The predictions (1 for spam, 0 for not spam)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if isinstance(X, str):
            X = [X]
        
        # Clean the data
        X_clean = [self.preprocess_text(text) for text in X]
        
        # Make predictions
        return self.model.predict(X_clean)
    
    def predict_proba(self, X):
        """
        Predict the probability of an email being spam
        
        Parameters:
        -----------
        X : str or list
            The email(s) to classify
            
        Returns:
        --------
        array-like
            The probability of each email being spam
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if isinstance(X, str):
            X = [X]
        
        # Clean the data
        X_clean = [self.preprocess_text(text) for text in X]
        
        # Make predictions
        return self.model.predict_proba(X_clean)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Parameters:
        -----------
        X_test : array-like
            The test data
        y_test : array-like
            The test labels
            
        Returns:
        --------
        dict
            A dictionary containing the accuracy, classification report, and confusion matrix
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Clean the data
        X_test_clean = [self.preprocess_text(text) for text in X_test]
        
        # Make predictions
        y_pred = self.model.predict(X_test_clean)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
    
    def save_model(self, filepath):
        """
        Save the model to a file
        
        Parameters:
        -----------
        filepath : str
            The path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load_model(cls, filepath, model_type='naive_bayes'):
        """
        Load a model from a file
        
        Parameters:
        -----------
        filepath : str
            The path to load the model from
        model_type : str
            The type of model to use
            
        Returns:
        --------
        SpamDetector
            A SpamDetector instance with the loaded model
        """
        detector = cls(model_type=model_type)
        
        with open(filepath, 'rb') as f:
            detector.model = pickle.load(f)
        
        return detector

# Example usage
if __name__ == "__main__":
    # Sample code to download a dataset and train the model
    import urllib.request
    import os
    import zipfile
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download spam/ham dataset (if it doesn't exist)
    dataset_path = 'data/spam_ham_dataset.csv'
    if not os.path.exists(dataset_path):
        print("Downloading dataset...")
        # You would replace this URL with a real dataset URL
        # For this example, we'll assume the file already exists or needs to be manually added
        # urllib.request.urlretrieve('URL_TO_DATASET', dataset_path)
        print("Please place a spam/ham dataset at", dataset_path)
        print("Format should be a CSV with 'text' and 'label' columns")
        print("Sample rows can be downloaded from sources like Kaggle")
        exit()
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Example: If the dataset has 'text' and 'label' columns
    # where 'label' is 'spam' or 'ham'
    if 'text' in df.columns and 'label' in df.columns:
        # Convert labels to binary (1 for spam, 0 for ham)
        df['binary_label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'].values, 
            df['binary_label'].values, 
            test_size=0.2, 
            random_state=42
        )
        
        # Create and train the model
        print("Training Naive Bayes model...")
        nb_detector = SpamDetector(model_type='naive_bayes')
        nb_detector.train(X_train, y_train)
        
        # Evaluate the model
        print("Evaluating Naive Bayes model...")
        nb_results = nb_detector.evaluate(X_test, y_test)
        print(f"Accuracy: {nb_results['accuracy']:.4f}")
        print("Classification Report:")
        print(nb_results['classification_report'])
        
        # Save the model
        print("Saving model...")
        if not os.path.exists('models'):
            os.makedirs('models')
        nb_detector.save_model('models/spam_detector_nb.pkl')
        
        # Optional: Train an SVM model
        print("Training SVM model...")
        svm_detector = SpamDetector(model_type='svm')
        svm_detector.train(X_train, y_train)
        
        # Evaluate the SVM model
        print("Evaluating SVM model...")
        svm_results = svm_detector.evaluate(X_test, y_test)
        print(f"Accuracy: {svm_results['accuracy']:.4f}")
        print("Classification Report:")
        print(svm_results['classification_report'])
        
        # Save the SVM model
        svm_detector.save_model('models/spam_detector_svm.pkl')
        
        # Example classification
        example_emails = [
            "Congratulations! You've won a free iPhone. Click here to claim your prize now!",
            "Hi John, just checking if we're still on for the meeting tomorrow at 2pm. Thanks, Sarah"
        ]
        
        print("\nExample classifications (Naive Bayes):")
        predictions = nb_detector.predict(example_emails)
        probabilities = nb_detector.predict_proba(example_emails)
        
        for i, email in enumerate(example_emails):
            print(f"Email: {email}")
            print(f"Prediction: {'Spam' if predictions[i] == 1 else 'Not Spam'}")
            print(f"Probability of being spam: {probabilities[i]:.4f}\n")
    else:
        print("Dataset format not recognized. Please ensure it has 'text' and 'label' columns.")