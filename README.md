# Data Preprocessing & Feature Engineering

A Python data-processing pipeline that cleans raw CSV data, handles missing values and duplicates, identifies date fields, engineers new features, and exports a processed dataset using pandas.

## Overview

Real-world datasets often need significant cleanup before they can be used for analysis or machine-learning workflows.

This project demonstrates a reusable preprocessing pipeline that takes raw CSV data and performs several common preparation and feature-engineering tasks automatically.

## Features

- Loads CSV data using pandas
- Standardizes column names
- Removes duplicate records
- Detects blank and missing values
- Fills missing numeric values using the median
- Fills missing categorical values using the mode
- Detects possible date and time columns
- Extracts year, month, day, and day-of-week features
- Generates numeric aggregate features
- Creates interaction features from numeric columns
- Exports the processed dataset to a new CSV file
- Displays before-and-after dataset information in the terminal

## Processing Workflow

The program follows a structured pipeline:

1. Load the original CSV dataset
2. Standardize column names
3. Remove duplicate data
4. Handle missing values
5. Detect and convert date fields
6. Engineer additional features
7. Compare the original and processed datasets
8. Save the processed data to a new CSV file

## Technologies

- Python
- pandas
- pathlib
- CSV data processing
- Data cleaning
- Feature engineering

## Project Structure

```text
data-preprocessing-feature-engineering/
│
├── data_preprocessing.py
├── original_data.csv
├── preprocessed_data.csv
├── requirements.txt
└── README.md
```
## Running the Project

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/joeyfarruggia/data-preprocessing-feature-engineering.git
cd data-preprocessing-feature-engineering
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the program

```bash
python data_preprocessing.py
```

The program reads `original_data.csv`, processes the dataset, and saves the completed output to `preprocessed_data.csv`.
