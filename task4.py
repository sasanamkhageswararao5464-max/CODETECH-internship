import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen
import warnings

warnings.filterwarnings('ignore')

# Create sample dataset for spam detection
def create_sample_spam_dataset():
    """
    Create a sample email dataset for spam detection
    """
    spam_emails = [
        "Click here to win FREE MONEY!",
        "You have won a lottery prize!",
        "Congratulations! You're a winner!",
        "Act now! Limited time offer!",
        "URGENT: Verify your account",
        "You've been selected! Click here",
        "Make money fast! Work from home",
        "Free gift card inside!",
        "Don't miss out! Call NOW!",
        "CLAIM YOUR PRIZE TODAY",
    ] * 10  # Repeat to increase dataset size

    ham_emails = [
        "Hi John, how are you doing?",
        "Let's meet for coffee tomorrow",
        "The meeting has been rescheduled",
        "Thanks for your help on the project",
        "Can you review the document?",
        "I'll send you the files soon",
        "Good morning! Ready for the presentation?",
        "The deadline has been extended",
        "Let me know about the event",
        "See you at the conference next week",
    ] * 10  # Repeat to increase dataset size

    # Create DataFrame
    emails = spam_emails + ham_emails
    labels = [1] * len(spam_emails) + [0] * len(ham_emails)  # 1 = spam, 0 = ham

    df = pd.DataFrame({
        'email': emails,
        'label': labels
    })

    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv('spam_emails.csv', index=False)
    return df

# Load and preprocess data
def load_and_preprocess_data(csv_file):
    """
    Load email data and split into train/test sets
    """
    df = pd.read_csv(csv_file)

    print("Dataset Information:")
    print(f"Total emails: {len(df)}")
    print(f"Spam emails: {df[df['label'] == 1].shape[0]}")
    print(f"Ham emails: {df[df['label'] == 0].shape[0]}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")

    # Split data into training and testing sets
    X = df['email']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

# Feature extraction using TF-IDF
def extract_features(X_train, X_test):
    """
    Convert text data to numerical features using TF-IDF
    """
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer

# Train Naive Bayes model
def train_naive_bayes(X_train, y_train):
    """
    Train Multinomial Naive Bayes classifier
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# Train Random Forest model
def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], zero_division=0))

    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

# Visualize results
def visualize_results(results_list, y_test):
    """
    Create visualizations of model performance
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Spam Email Detection - Model Performance Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Accuracy Comparison
    models = [r['model'] for r in results_list]
    accuracies = [r['accuracy'] for r in results_list]

    ax1 = axes[0, 0]
    ax1.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

    # Plot 2: Metrics Comparison
    ax2 = axes[0, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for result in results_list:
        values = [result['accuracy'], result['precision'], result['recall'], result['f1_score']]
        ax2.plot(metrics, values, marker='o', label=result['model'], linewidth=2, markersize=8)

    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('Metrics Comparison Across Models')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])

    # Plot 3: Confusion Matrix for Naive Bayes
    ax3 = axes[1, 0]
    cm_nb = results_list[0]['confusion_matrix']
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=True,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax3.set_title(f"Confusion Matrix - {results_list[0]['model']}")
    ax3.set_ylabel('True Label', fontweight='bold')
    ax3.set_xlabel('Predicted Label', fontweight='bold')

    # Plot 4: Confusion Matrix for Random Forest
    ax4 = axes[1, 1]
    cm_rf = results_list[1]['confusion_matrix']
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax4, cbar=True,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax4.set_title(f"Confusion Matrix - {results_list[1]['model']}")
    ax4.set_ylabel('True Label', fontweight='bold')
    ax4.set_xlabel('Predicted Label', fontweight='bold')

    plt.tight_layout()
    plt.savefig('spam_detection_results.png', dpi=300, bbox_inches='tight')
    print("✓ Results visualization saved as 'spam_detection_results.png'")
    plt.show()

# Predict on new emails
def predict_spam(model, vectorizer, email_text):
    """
    Predict if an email is spam or ham
    """
    email_tfidf = vectorizer.transform([email_text])
    prediction = model.predict(email_tfidf)[0]
    confidence = model.predict_proba(email_tfidf)[0]

    result = "SPAM" if prediction == 1 else "HAM"
    confidence_score = max(confidence)

    print(f"\nEmail: {email_text[:50]}...")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence_score:.2%}\n")

    return result

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("SPAM EMAIL DETECTION - MACHINE LEARNING MODEL")
    print("="*60 + "\n")

    # Create dataset
    print("Creating sample dataset...")
    df = create_sample_spam_dataset()
    print("✓ Dataset created\n")

    # Load and preprocess
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data('spam_emails.csv')
    print("✓ Data preprocessed\n")

    # Extract features
    print("Extracting TF-IDF features...")
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)
    print("✓ Features extracted\n")

    # Train models
    print("Training Naive Bayes model...")
    nb_model = train_naive_bayes(X_train_tfidf, y_train)
    print("✓ Naive Bayes trained\n")

    print("Training Random Forest model...")
    rf_model = train_random_forest(X_train_tfidf, y_train)
    print("✓ Random Forest trained\n")

    # Evaluate models
    results = []
    results.append(evaluate_model(nb_model, X_test_tfidf, y_test, "Naive Bayes"))
    results.append(evaluate_model(rf_model, X_test_tfidf, y_test, "Random Forest"))

    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(results, y_test)

    # Test with new emails
    print("\n" + "="*60)
    print("Testing with new emails")
    print("="*60 + "\n")

    test_emails = [
        "Hi, meeting tomorrow at 10 AM",
        "Click here NOW to win FREE cash!!!",
        "Project deadline extended by one week",
        "URGENT: Verify your account immediately!"
    ]

    for email in test_emails:
        predict_spam(nb_model, vectorizer, email)
