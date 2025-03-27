import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def load_data():
    """Load the Titanic dataset."""
    train_data = pd.read_csv("../data/train.csv")
    test_data = pd.read_csv("../data/test.csv")
    return train_data, test_data

def preprocess_data(df):
    """Preprocess the data by handling missing values and encoding categorical variables."""
    # Create a copy of the dataframe
    df = df.copy()
    
    # Fill missing Age with median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Fill missing Fare with median
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Fill missing Embarked with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Convert categorical variables to numeric
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Drop unnecessary columns
    columns_to_drop = ['Cabin', 'Ticket', 'Name', 'PassengerId']
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    return df

def train_model(X_train, y_train):
    """Train the logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_mat)
    print("\nClassification Report:")
    print(report)
    
    return y_pred

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': abs(model.coef_[0])
    })
    importance = importance.sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance['feature'], importance['importance'])
    plt.title('Feature Importance')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.savefig('../notebooks/feature_importance.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    train_data, test_data = load_data()
    
    # Preprocess data
    print("Preprocessing data...")
    train_processed = preprocess_data(train_data)
    test_processed = preprocess_data(test_data)
    
    # Split features and target
    X = train_processed.drop('Survived', axis=1)
    y = train_processed['Survived']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training model...")
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = evaluate_model(model, X_test_scaled, y_test)
    
    # Plot feature importance
    print("\nPlotting feature importance...")
    plot_feature_importance(model, X.columns)

if __name__ == "__main__":
    main() 