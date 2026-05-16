#!/usr/bin/env python3
"""Pipeline Part 2: Use labeled examples to find similar words via machine learning.

This script takes the output from pipeline.py and uses user-provided examples
to train a classifier that finds other similar words in the dataset.

Usage:
    python pipeline_part2.py <input_csv> <example_indices> [options]

Steps:
  1. Load the CSV output from pipeline.py
  2. Use user-provided row indices as positive examples (words matching your criteria)
  3. Extract features from the top N rows (default 200) for training
  4. Train a binary classifier on those rows
  5. Classify ALL words in the dataset as positive or negative
  6. Output all positive classifications to CSV (sorted by confidence)

Example:
    python pipeline_part2.py output.csv --examples 0 5 12 42
    # Uses rows 0, 5, 12, 42 as positive examples from top 200 rows
    # Classifies entire dataset, outputs all positive words
"""

import argparse
import sys
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV output from pipeline.py.

    Parameters:
        csv_path: Path to the CSV file from pipeline.py

    Returns:
        DataFrame with all computed features
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} words from {csv_path}")
    return df


def create_feature_matrix(
    df: pd.DataFrame,
    feature_columns: List[str] = None,
) -> np.ndarray:
    """Create a feature matrix from the DataFrame for ML model.

    Parameters:
        df: Input DataFrame with computed features
        feature_columns: List of column names to use as features.
                        If None, uses default similarity/metric columns.

    Returns:
        2D numpy array of shape (n_samples, n_features)
    """
    if feature_columns is None:
        # Default features: use all the computed metrics
        feature_columns = [
            "cosine_distance",
            "definition_word_count",
            "orthographic_similarity",
            "cognate_similarity",
        ]
        # Add concreteness if it exists (optional feature)
        if "concreteness" in df.columns:
            feature_columns.append("concreteness")

    # Extract features and handle missing values
    features = df[feature_columns].fillna(0.0).values
    return features


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "logistic_regression",
) -> tuple:
    """Train a binary classifier on the features and labels.

    Parameters:
        X: Feature matrix (n_samples, n_features)
        y: Binary labels (1 for positive examples, 0 for others)
        model_type: Type of classifier to use.
                   Options: "logistic_regression", "svm", "random_forest"

    Returns:
        Tuple of (classifier, scaler)
    """
    # Scale features for better ML performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the appropriate classifier
    if model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "svm":
        clf = SVC(kernel="rbf", probability=True, random_state=42)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    clf.fit(X_scaled, y)
    return (clf, scaler)


def score_words(
    X: np.ndarray,
    clf: object,
    scaler: object,
) -> np.ndarray:
    """Score all words using the trained classifier.

    Higher scores indicate higher similarity to the positive examples.

    Parameters:
        X: Feature matrix for all words
        clf: Trained classifier
        scaler: Fitted StandardScaler

    Returns:
        1D array of similarity scores (0-1)
    """
    X_scaled = scaler.transform(X)
    # Get probability of being in the positive class (label 1)
    scores = clf.predict_proba(X_scaled)[:, 1]
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify words and find all that are similar to user-provided examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify all words, output only positive classifications
  python pipeline_part2.py output.csv --examples 0 5 12

  # Use SVM classifier (often more accurate)
  python pipeline_part2.py output.csv --examples 10 20 30 --model svm

  # Train on top 500 words
  python pipeline_part2.py output.csv --examples 1 2 3 --train-size 500 --model random_forest

  # Limit positive results to top 100 (by confidence score)
  python pipeline_part2.py output.csv --examples 0 5 --top-n 100
""",
    )
    parser.add_argument("input_csv", help="Path to the CSV from pipeline.py")
    parser.add_argument(
        "--examples",
        type=int,
        nargs="+",
        required=True,
        help="Row indices (0-based) of words that match your criteria. Example: --examples 0 5 12",
    )
    parser.add_argument(
        "--output",
        default="similar_words.csv",
        help="Output CSV path (default: similar_words.csv)",
    )
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "svm", "random_forest"],
        default="logistic_regression",
        help="ML model to use (default: logistic_regression)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=200,
        help="Number of top-ranked words to use for training (default: 200)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only output top N positive classifications. If not specified, outputs all positive words.",
    )

    args = parser.parse_args()

    # Step 1: Load data
    df = load_and_prepare_data(args.input_csv)

    # Step 2: Validate example indices
    example_indices = args.examples
    if not example_indices:
        print("Error: Must provide at least one example index with --examples")
        sys.exit(1)

    invalid_indices = [i for i in example_indices if i < 0 or i >= len(df)]
    if invalid_indices:
        print(f"Error: Invalid row indices: {invalid_indices}")
        print(f"Valid range: 0 to {len(df) - 1}")
        sys.exit(1)

    # Print example words
    print(f"\nUsing {len(example_indices)} example(s):")
    for idx in example_indices:
        word = df.iloc[idx]["From-Language word"]
        definition = df.iloc[idx]["To-Language definition"]
        print(f"  [{idx}] {word}: {definition}")

    # Step 3: Split data - train on top N rows, classify all
    print(f"\nTraining on top {args.train_size} rows...")
    df_train = df.head(args.train_size).copy()
    df_all = df.copy()

    # Validate that examples are within training set
    examples_in_train = [i for i in example_indices if i < args.train_size]
    examples_not_in_train = [i for i in example_indices if i >= args.train_size]
    if examples_not_in_train:
        print(
            f"Warning: Examples {examples_not_in_train} are outside training set (rows 0-{args.train_size - 1})."
        )
        if not examples_in_train:
            print(
                f"Error: No examples in training set. Please provide examples from rows 0-{args.train_size - 1}."
            )
            sys.exit(1)
        print(f"Using only examples {examples_in_train} for training.")
        example_indices = examples_in_train

    # Step 4: Create feature matrices
    print("Extracting features...")
    X_train = create_feature_matrix(df_train)
    X_all = create_feature_matrix(df_all)

    # Step 5: Create binary labels for training set (1 for positive examples, 0 otherwise)
    y_train = np.zeros(len(df_train), dtype=int)
    y_train[example_indices] = 1
    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    print(
        f"Training set: {n_positive} positive example(s), {n_negative} negative examples"
    )

    # Step 6: Train classifier on training set only
    print(f"\nTraining {args.model} classifier on top {args.train_size} rows...")
    clf, scaler = train_classifier(X_train, y_train, model_type=args.model)
    print("Classifier trained.")

    # Step 7: Classify ALL words (including those outside training set)
    print(f"\nClassifying all {len(df_all)} words in dataset...")
    similarity_scores = score_words(X_all, clf, scaler)
    predictions = clf.predict(scaler.transform(X_all))

    df_all["similarity_score"] = similarity_scores
    df_all["prediction"] = predictions

    # Step 8: Filter to only positive classifications
    df_positive = df_all[df_all["prediction"] == 1].copy()
    df_positive = df_positive.sort_values(
        "similarity_score", ascending=False
    ).reset_index(drop=True)

    n_positive_found = len(df_positive)
    n_total = len(df_all)
    n_negative_found = n_total - n_positive_found

    print(
        f"Classification complete: {n_positive_found} positive, {n_negative_found} negative out of {n_total} total"
    )

    if n_positive_found == 0:
        print("\nWarning: No words classified as positive. Try:")
        print("  - Using different examples")
        print("  - Increasing --train-size")
        print("  - Using a different classifier (--model svm or --model random_forest)")
        sys.exit(0)

    # Step 9: Optionally limit to top N (if specified)
    if args.top_n is not None:
        if args.top_n < len(df_positive):
            print(f"Limiting output to top {args.top_n} positive words.")
            df_positive = df_positive.head(args.top_n)

    # Step 10: Output results
    output_columns = [
        "From-Language word",
        "To-Language definition",
        "similarity_score",
        "cosine_distance",
        "definition_word_count",
        "orthographic_similarity",
        "cognate_similarity",
    ]
    # Only include columns that exist
    available_columns = [col for col in output_columns if col in df_positive.columns]

    df_positive[available_columns].to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print(f"Saved {len(df_positive)} positive words to output\n")

    # Print top 10 results
    print("Top 10 positive results:")
    print("-" * 120)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", None)
    print(df_positive[available_columns].head(10).to_string(index=True))
    print("-" * 120)


if __name__ == "__main__":
    main()
