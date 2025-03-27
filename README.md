# Titanic Survival Prediction

This project uses machine learning to predict passenger survival on the Titanic based on historical data.

## Project Overview

The goal of this project is to build a predictive model that can determine whether a passenger would have survived the Titanic disaster based on various features such as:
- Passenger class
- Age
- Sex
- Fare
- Family size
- Port of embarkation

## Project Structure

```
.
├── data/               # Directory for storing the dataset
├── notebooks/         # Jupyter notebooks for exploratory data analysis
├── src/              # Source code
│   ├── data.py       # Data loading and preprocessing
│   ├── model.py      # Model training and evaluation
│   └── utils.py      # Utility functions
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Titanic dataset:
   - Visit [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
   - Download the train.csv and test.csv files
   - Place them in the `data/` directory

## Usage

1. Run the main script:
```bash
python src/main.py
```

2. Or explore the data using the Jupyter notebook:
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## Model Performance

The model's performance will be evaluated using:
- Accuracy score
- Precision, Recall, and F1-score
- Confusion matrix
- ROC curve and AUC score

## Contributing

Feel free to submit issues and enhancement requests! 