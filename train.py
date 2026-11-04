"""Train the product-name classifier and report cross-validated errors."""

import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

HERE         = Path( __file__ ).parent
INPUT_FILE   = HERE / "data" / "products.xlsx"
MODEL_FILE   = HERE / "model.joblib"
NAME_COLUMN  = "product_name"
LABEL_COLUMN = "is_shoe"

# AM 20/Oct/26 - set True when the label column means the opposite, for example "not shoes"
INVERT_LABEL = False

EXAMPLES_TO_SHOW = 5
THRESHOLD        = 0.5
TEST_ROWS        = 20_000


def fix_encoding( value ) -> str:
    # AM 20/Oct/26 - repairs UTF-8 bytes that were decoded as cp1252: "MÃ¤dchen" becomes "Mädchen"
    text = str( value )
    try:
        return text.encode( "cp1252" ).decode( "utf-8" )
    except UnicodeError:
        return text


def normalize_name( value ) -> str:
    return fix_encoding( value ).lower()


def build_model():
    # AM 20/Oct/26 - character n-grams handle compound words, typos, truncation and several languages
    return make_pipeline(
        TfidfVectorizer( analyzer="char_wb", ngram_range=( 2, 5 ), min_df=2, sublinear_tf=True ),
        LogisticRegression( max_iter=2000, C=10, class_weight="balanced" ),
    )


def load_unique( input_file: Path, nrows: int | None ) -> pd.DataFrame:
    df = pd.read_excel( input_file, usecols=[ NAME_COLUMN, LABEL_COLUMN ], nrows=nrows )
    # AM 20/Oct/26 - the same name repeats many times; duplicates would leak between CV folds
    return df.drop_duplicates( NAME_COLUMN ).reset_index( drop=True ), len( df )


def print_examples( title: str, rows: pd.DataFrame, probability_column: str, limit: int ) -> None:
    print( f"\n{title}: {len(rows)} total, showing {min(limit, len(rows))}" )
    for _, row in rows.head( limit ).iterrows():
        print( f"  p(positive)={row[probability_column]:.2f}  {fix_encoding(row[NAME_COLUMN])}" )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser( description="Train the product-name classifier." )
    parser.add_argument( "--input", type=Path, default=INPUT_FILE, help="Excel file with names and labels" )
    parser.add_argument( "--test", action="store_true", help=f"read only the first {TEST_ROWS} rows and skip saving" )
    parser.add_argument( "--examples", type=int, default=EXAMPLES_TO_SHOW, help="how many error examples to print" )
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure( encoding="utf-8" )
    args    = parse_args()
    started = time.perf_counter()
    if args.test:
        print( f"TEST MODE: first {TEST_ROWS} rows, the model will not be saved" )

    unique, total_rows = load_unique( args.input, TEST_ROWS if args.test else None )
    names              = unique[ NAME_COLUMN ].map( normalize_name )
    labels             = unique[ LABEL_COLUMN ]
    target             = 1 - labels if INVERT_LABEL else labels

    cv          = StratifiedKFold( n_splits=5, shuffle=True, random_state=0 )
    probability = cross_val_predict( build_model(), names, target, cv=cv, method="predict_proba" )[ :, 1 ]
    predicted   = ( probability >= THRESHOLD ).astype( int )

    print( f"Unique names: {len(unique)} (rows: {total_rows})" )
    print( classification_report( target, predicted, digits=3, target_names=[ "negative", "positive" ] ) )

    # AM 20/Oct/26 - most confident errors first: these are the likeliest labeling mistakes in the source
    unique[ "probability" ] = probability
    false_positives         = unique[ ( predicted == 1 ) & ( target == 0 ) ].sort_values( "probability", ascending=False )
    false_negatives         = unique[ ( predicted == 0 ) & ( target == 1 ) ].sort_values( "probability" )
    print_examples( "Model says positive, label says negative", false_positives, "probability", args.examples )
    print_examples( "Model says negative, label says positive", false_negatives, "probability", args.examples )

    if args.test:
        print( "\nTest mode: model not saved" )
    else:
        model = build_model().fit( names, target )
        joblib.dump( model, MODEL_FILE )
        print( f"\nModel saved to {MODEL_FILE}" )

    print( f"Done in {( time.perf_counter() - started ) / 60:.2f} minutes" )


if __name__ == "__main__":
    main()
