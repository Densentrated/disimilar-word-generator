#!/usr/bin/env python3
"""Pipeline Part 2: Use labeled examples to find similar words via machine learning.

This script takes the output from pipeline.py and uses user-provided examples
to train a classifier that finds other similar words in the dataset.

Usage:
    python pipeline_part2.py <input_csv> <example_indices> [options]

Steps:
  1. Load the CSV output from pipeline.py
  2. Use user-provided row indices as positive examples (words matching your criteria)
  3. Extract features from those examples
  4. Train a binary classifier (e.g., logistic regression, SVM, or random forest)
  5. Score all words in the dataset
  6. Output a ranked CSV of words sorted by similarity to your examples

Example:
    python pipeline_part2.py output.csv --examples 0 5 12 42
    # Uses rows 0, 5, 12, 42 as positive examples
    # Finds other words similar to those
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
) -> object:
    """Train a binary classifier on the features and labels.

    Parameters:
        X: Feature matrix (n_samples, n_features)
        y: Binary labels (1 for positive examples, 0 for others)
        model_type: Type of classifier to use.
                   Options: "logistic_regression", "svm", "random_forest"

    Returns:
        Trained classifier object
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
        description="Find words similar to user-provided examples using ML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find words similar to rows 0, 5, and 12 (indices from the CSV)
  python pipeline_part2.py output.csv --examples 0 5 12

  # Use SVM classifier instead of logistic regression
  python pipeline_part2.py output.csv --examples 10 20 30 --model svm

  # Use random forest and output to custom file
  python pipeline_part2.py output.csv --examples 1 2 3 --output similar_words.csv --model random_forest
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
        "--top-n",
        type=int,
        default=None,
        help="Only output top N similar words. If not specified, outputs all words.",
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

    # Step 3: Create feature matrix
    print("\nExtracting features...")
    X = create_feature_matrix(df)

    # Step 4: Create binary labels (1 for positive examples, 0 otherwise)
    y = np.zeros(len(df), dtype=int)
    y[example_indices] = 1
    print(f"Training set: {y.sum()} positive examples, {len(y) - y.sum()} negative examples")

    # Step 5: Train classifier
    print(f"\nTraining {args.model} classifier...")
    clf, scaler = train_classifier(X, y, model_type=args.model)
    print("Classifier trained.")

    # Step 6: Score all words
    print("Scoring all words...")
    similarity_scores = score_words(X, clf, scaler)

    # Step 7: Add scores to DataFrame and sort
    df["similarity_score"] = similarity_scores
    df_sorted = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)

    # Step 8: Optionally limit to top N
    if args.top_n is not None:
        df_sorted = df_sorted.head(args.top_n)
        print(f"Keeping top {args.top_n} similar words.")

    # Step 9: Output results
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
    available_columns = [col for col in output_columns if col in df_sorted.columns]

    df_sorted[available_columns].to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print(f"Found {len(df_sorted)} similar words (sorted by similarity score)\n")

    # Print top 10 results
    print("Top 10 results:")
    print("-" * 120)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", None)
    print(df_sorted[available_columns].head(10).to_string(index=True))
    print("-" * 120)


if __name__ == "__main__":
    main()
