# example.py
# Example script showing how to use the Spam Detector

from spam_detector import SpamDetector
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    print("Spam Email Detector Example")
    print("==========================\n")

    # Check if we have a dataset
    dataset_path = 'data/spam_ham_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Downloading a sample dataset...")
        
        # This is a placeholder - in a real implementation, you'd download a dataset
        # For this example, we'll create a tiny sample dataset
        sample_data = {
            'text': [
                "Congratulations! You've won $1000! Claim now!",
                "URGENT: Your account has been compromised. Click here!",
                "Hi John, can we meet tomorrow at 2pm?",
                "The quarterly report is attached for your review",
                "FREE VIAGRA! Best prices guaranteed!",
                "Meeting agenda for next week's planning session",
                "Your package will be delivered tomorrow between 9-5",
                "ALERT: Your bank account requires verification immediately"
            ],
            'label': ['spam', 'spam', 'ham', 'ham', 'spam', 'ham', 'ham', 'spam']
        }
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save sample data
        pd.DataFrame(sample_data).to_csv(dataset_path, index=False, encoding='utf-8')
        print(f"Created sample dataset at {dataset_path}")

    # Load the dataset
    print("Loading dataset...")
    try:
        # Try reading with UTF-8 encoding first
        df = pd.read_csv(dataset_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If that fails, try with a more lenient encoding
        df = pd.read_csv(dataset_path, encoding='latin1')
    
    # Convert labels to binary (1 for spam, 0 for ham)
    df['binary_label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)
    
    # Split the data
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, 
        df['binary_label'].values, 
        test_size=0.2, 
        random_state=42
    )
    
    # Create and train the model
    print("\nTraining Naive Bayes model...")
    detector = SpamDetector(model_type='naive_bayes')
    detector.train(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating model performance:")
    results = detector.evaluate(X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save the model
    print("\nSaving model...")
    os.makedirs('models', exist_ok=True)
    model_path = 'models/spam_detector.pkl'
    detector.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Test with new examples
    print("\nTesting with new examples:")
    test_emails = [
        "URGENT: Your account has been suspended. Verify now to restore access!",
        "Hey Sarah, just checking if we're still on for lunch tomorrow?",
        "CONGRATULATIONS! You've been selected to receive a free gift card!",
        "The quarterly sales report is attached for your review. Please provide feedback."
    ]
    
    # Make predictions
    predictions = detector.predict(test_emails)
    probabilities = detector.predict_proba(test_emails)
    
    print("\nResults:")
    print("-" * 60)
    for i, email in enumerate(test_emails):
        print(f"Email: {email}")
        print(f"Prediction: {'Spam' if predictions[i] == 1 else 'Not Spam'}")
        print(f"Probability of being spam: {probabilities[i]:.4f}")
        print("-" * 60)
    
    # Load the model example
    print("\nDemonstrating model loading:")
    loaded_detector = SpamDetector.load_model(model_path)
    loaded_prediction = loaded_detector.predict([test_emails[0]])[0]
    loaded_probability = loaded_detector.predict_proba([test_emails[0]])[0]
    
    print(f"Loaded model prediction: {'Spam' if loaded_prediction == 1 else 'Not Spam'}")
    print(f"Loaded model probability: {loaded_probability:.4f}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()