"""
Data Preprocessing and Feature Engineering Pipeline

This program:
1. Loads a CSV dataset using pandas.
2. Cleans the dataset by handling missing values and duplicates.
3. Engineers new features from the existing data.
4. Saves the processed dataset as a new CSV file.
"""

from pathlib import Path

import pandas as pd


# File locations
PROJECT_FOLDER = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_FOLDER / "original_data.csv"
OUTPUT_FILE = PROJECT_FOLDER / "preprocessed_data.csv"


def load_data(file_path: Path) -> pd.DataFrame:
    """Load a CSV file and return it as a pandas DataFrame."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find the dataset at:\n{file_path}\n\n"
            "Make sure the CSV is in the same folder as this Python file "
            "and is named original_data.csv."
        )

    dataframe = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print(f"Original rows: {dataframe.shape[0]}")
    print(f"Original columns: {dataframe.shape[1]}")

    return dataframe


def clean_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to a consistent lowercase format."""

    dataframe = dataframe.copy()

    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    return dataframe


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset.

    Cleaning operations:
    - Standardize column names.
    - Remove duplicate rows.
    - Replace blank strings with missing values.
    - Fill missing numeric values with the column median.
    - Fill missing categorical values with the column mode.
    """

    cleaned_data = clean_column_names(dataframe)

    rows_before_duplicates = len(cleaned_data)
    cleaned_data = cleaned_data.drop_duplicates()
    duplicates_removed = rows_before_duplicates - len(cleaned_data)

    print(f"\nDuplicate rows removed: {duplicates_removed}")

    # Convert empty or whitespace-only strings into missing values.
    object_columns = cleaned_data.select_dtypes(include=["object", "string"]).columns

    for column in object_columns:
        cleaned_data[column] = cleaned_data[column].replace(
            r"^\s*$",
            pd.NA,
            regex=True,
        )

    missing_before = int(cleaned_data.isna().sum().sum())
    print(f"Missing values before cleaning: {missing_before}")

    # Fill missing numeric values with each column's median.
    numeric_columns = cleaned_data.select_dtypes(include="number").columns

    for column in numeric_columns:
        median_value = cleaned_data[column].median()
        cleaned_data[column] = cleaned_data[column].fillna(median_value)

    # Fill missing text/categorical values with each column's most common value.
    categorical_columns = cleaned_data.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:
        mode_values = cleaned_data[column].mode(dropna=True)

        if not mode_values.empty:
            replacement_value = mode_values.iloc[0]
        else:
            replacement_value = "Unknown"

        cleaned_data[column] = cleaned_data[column].fillna(replacement_value)

    missing_after = int(cleaned_data.isna().sum().sum())
    print(f"Missing values after cleaning: {missing_after}")

    return cleaned_data


def convert_possible_date_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to convert columns containing 'date' or 'time' in their names
    into pandas datetime values.
    """

    converted_data = dataframe.copy()

    possible_date_columns = [
        column
        for column in converted_data.columns
        if "date" in column.lower() or "time" in column.lower()
    ]

    for column in possible_date_columns:
        converted_dates = pd.to_datetime(
            converted_data[column],
            errors="coerce",
        )

        # Only use the conversion when at least half of the values
        # were successfully recognized as dates.
        successful_conversion_rate = converted_dates.notna().mean()

        if successful_conversion_rate >= 0.50:
            converted_data[column] = converted_dates
            print(f"Recognized date column: {column}")

    return converted_data


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from the existing dataset.

    New features may include:
    - Year, month, day, and day of week from date columns.
    - Sum, average, and range of numeric columns.
    - An interaction term created by multiplying two numeric columns.
    """

    engineered_data = convert_possible_date_columns(dataframe)

    original_columns = list(engineered_data.columns)

    # Create features from datetime columns.
    date_columns = engineered_data.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns

    for column in date_columns:
        engineered_data[f"{column}_year"] = engineered_data[column].dt.year
        engineered_data[f"{column}_month"] = engineered_data[column].dt.month
        engineered_data[f"{column}_day"] = engineered_data[column].dt.day
        engineered_data[f"{column}_day_of_week"] = (
            engineered_data[column].dt.day_name()
        )

    # Locate the original numeric columns.
    numeric_columns = list(
        engineered_data[original_columns]
        .select_dtypes(include="number")
        .columns
    )

    if numeric_columns:
        engineered_data["numeric_feature_sum"] = (
            engineered_data[numeric_columns].sum(axis=1)
        )

        engineered_data["numeric_feature_average"] = (
            engineered_data[numeric_columns].mean(axis=1)
        )

        engineered_data["numeric_feature_range"] = (
            engineered_data[numeric_columns].max(axis=1)
            - engineered_data[numeric_columns].min(axis=1)
        )

    # Create an interaction term when at least two numeric features exist.
    if len(numeric_columns) >= 2:
        first_column = numeric_columns[0]
        second_column = numeric_columns[1]

        interaction_column_name = (
            f"{first_column}_x_{second_column}"
        )

        engineered_data[interaction_column_name] = (
            engineered_data[first_column]
            * engineered_data[second_column]
        )

        print(
            "Created interaction feature: "
            f"{interaction_column_name}"
        )

    new_columns = [
        column
        for column in engineered_data.columns
        if column not in original_columns
    ]

    print(f"\nNew engineered features: {len(new_columns)}")

    for column in new_columns:
        print(f"  - {column}")

    return engineered_data


def save_data(dataframe: pd.DataFrame, file_path: Path) -> None:
    """Save the processed DataFrame as a new CSV file."""

    dataframe.to_csv(file_path, index=False)

    print("\nProcessed dataset saved successfully.")
    print(f"Saved to: {file_path}")


def compare_datasets(
    original_data: pd.DataFrame,
    processed_data: pd.DataFrame,
) -> None:
    """Display a basic comparison of the original and processed datasets."""

    print("\n" + "=" * 50)
    print("DATASET COMPARISON")
    print("=" * 50)

    print(
        f"Original dataset shape: "
        f"{original_data.shape[0]} rows, "
        f"{original_data.shape[1]} columns"
    )

    print(
        f"Processed dataset shape: "
        f"{processed_data.shape[0]} rows, "
        f"{processed_data.shape[1]} columns"
    )

    print(
        f"Original missing values: "
        f"{int(original_data.isna().sum().sum())}"
    )

    print(
        f"Processed missing values: "
        f"{int(processed_data.isna().sum().sum())}"
    )

    print("\nFirst five processed records:")
    print(processed_data.head())


def main() -> None:
    """Run the complete data preprocessing workflow."""

    try:
        original_data = load_data(INPUT_FILE)

        cleaned_data = clean_data(original_data)

        processed_data = engineer_features(cleaned_data)

        save_data(processed_data, OUTPUT_FILE)

        compare_datasets(original_data, processed_data)

    except FileNotFoundError as error:
        print(error)

    except pd.errors.EmptyDataError:
        print("The CSV file is empty.")

    except pd.errors.ParserError as error:
        print(f"Pandas could not properly read the CSV file: {error}")

    except Exception as error:
        print(f"An unexpected error occurred: {error}")


if __name__ == "__main__":
    main()
