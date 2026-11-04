"""Classify product names with the trained model and write the result to a new Excel file."""

import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from train import INPUT_FILE, LABEL_COLUMN, MODEL_FILE, NAME_COLUMN, THRESHOLD, TEST_ROWS, normalize_name, print_examples

REVIEW_BAND      = ( 0.2, 0.8 )
EXAMPLES_TO_SHOW = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser( description="Classify product names with the trained model." )
    parser.add_argument( "--input", type=Path, default=INPUT_FILE, help="Excel file with a product name column" )
    parser.add_argument( "--output", type=Path, default=None, help="where to write the result" )
    parser.add_argument( "--test", action="store_true", help=f"read only the first {TEST_ROWS} rows" )
    parser.add_argument( "--examples", type=int, default=EXAMPLES_TO_SHOW, help="how many disagreements to print" )
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure( encoding="utf-8" )
    args    = parse_args()
    started = time.perf_counter()
    output  = args.output or args.input.with_name( f"{args.input.stem} - classified.xlsx" )
    if args.test:
        print( f"TEST MODE: first {TEST_ROWS} rows" )

    model = joblib.load( MODEL_FILE )
    df    = pd.read_excel( args.input, nrows=TEST_ROWS if args.test else None )

    # AM 20/Oct/26 - predict once per unique name, then join back to all rows
    unique                   = df[ [ NAME_COLUMN ] ].drop_duplicates()
    unique[ "probability" ]  = model.predict_proba( unique[ NAME_COLUMN ].map( normalize_name ) )[ :, 1 ]
    unique[ "predicted" ]    = ( unique[ "probability" ] >= THRESHOLD ).astype( int )
    unique[ "needs_review" ] = unique[ "probability" ].between( *REVIEW_BAND ).astype( int )

    result = df.merge( unique, on=NAME_COLUMN, how="left" )
    result.to_excel( output, index=False )

    print( f"Rows: {len(result)}, unique names: {len(unique)}" )
    print( f"Positive: {unique['predicted'].sum()} names, needs review: {unique['needs_review'].sum()} names" )
    print( f"Saved to {output}" )

    # AM 20/Oct/26 - the model was trained on these labels, so expect fewer disagreements than in train.py
    if LABEL_COLUMN in result.columns:
        report = result.drop_duplicates( NAME_COLUMN )
        missed = report[ ( report[ LABEL_COLUMN ] == 1 ) & ( report[ "predicted" ] == 0 ) ].sort_values( "probability" )
        found  = report[ ( report[ LABEL_COLUMN ] == 0 ) & ( report[ "predicted" ] == 1 ) ].sort_values( "probability", ascending=False )
        print_examples( "Label says positive, model says negative", missed, "probability", args.examples )
        print_examples( "Label says negative, model says positive", found, "probability", args.examples )

    print( f"\nDone in {( time.perf_counter() - started ) / 60:.2f} minutes" )


if __name__ == "__main__":
    main()
