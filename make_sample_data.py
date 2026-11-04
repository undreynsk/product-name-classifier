"""Generate a synthetic product catalog so the project can be run without real data."""

import random
from pathlib import Path

import pandas as pd

HERE            = Path( __file__ ).parent
OUTPUT_FILE     = HERE / "data" / "products.xlsx"
ROWS            = 40_000
SEED            = 20

# AM 04/Nov/26 - the source system cuts every name at 40 characters, so the sample does the same
# This is done to emulate the bad data converting (like if we cut at 40 symbols at some stage)
MAX_NAME_LENGTH = 40

SHOE_CATEGORIES     = [ "Sneaker", "Sandalen", "Stiefel", "Halbschuhe", "Slipper", "Pumps", "Sport-Schuhe" ]
NON_SHOE_CATEGORIES = [ "Jeans", "Hosen", "Kleider", "Shirts & Polos", "Taschen", "Jacken", "Röcke" ]
AUDIENCES           = [ "Damen", "Herren", "Jungen", "Mädchen", "Unisex" ]
BRANDS              = [ "Nordvik", "Lumio", "Castra", "Pelomar", "Avrina", "Tessano" ]
MODELS              = [ "CLASSIC LOW", "URBAN 2.0", "STREET RUN", "SOFT WALK", "CITY LINE", "DAILY FIT", "TRAIL PRO" ]

# AM 20/Oct/26 - traps on purpose: the words look like footwear but the product is not a shoe
TRAPS = [
    ( "Damen - Jeans 'BOOTCUT MEDIUM BLUE'", 0 ),
    ( "Herren - Jeans 'SLIM BOOTCUT RINSE'", 0 ),
    ( "Damen - Shirts & Tops 'U-Boot-Ausschnitt'", 0 ),
    ( "Unisex - Strümpfe & Socken 'Sneakersocken 3er Pack'", 0 ),
    ( "Damen - Accessoires 'Schuhbeutel Baumwolle'", 0 ),
    ( "Herren - Pflege 'Schuhcreme schwarz'", 0 ),
]


def make_name( rng: random.Random, is_shoe: bool ) -> str:
    category = rng.choice( SHOE_CATEGORIES if is_shoe else NON_SHOE_CATEGORIES )
    audience = rng.choice( AUDIENCES )
    style    = rng.choice( [ "structured", "brand", "plain" ] )
    if style == "structured":
        name = f"{audience} - {category} '{rng.choice( MODELS )}'"
    elif style == "brand":
        name = f"{rng.choice( BRANDS )} {category.lower()} {rng.choice( MODELS ).lower()}"
    else:
        name = f"{category} {rng.choice( MODELS ).title()}"
    return name


def truncate( name: str ) -> str:
    # AM 04/Nov/26 - applied to every name, generated and hand written alike
    return name[ :MAX_NAME_LENGTH ]


def main() -> None:
    rng  = random.Random( SEED )
    rows = []
    
    for _ in range( ROWS - len( TRAPS ) ):
        # AM 20/Oct/26 - roughly 7% shoes, like a real mixed catalog
        is_shoe = rng.random() < 0.07
        rows.append( { "product_name": truncate( make_name( rng, is_shoe ) ), "is_shoe": int( is_shoe ) } )

    rows += [ { "product_name": truncate( name ), "is_shoe": label } for name, label in TRAPS ]
    rng.shuffle( rows )

    df = pd.DataFrame( rows )
    OUTPUT_FILE.parent.mkdir( exist_ok=True )
    df.to_excel( OUTPUT_FILE, index=False )
    print( f"Wrote {len(df)} rows, {df['product_name'].nunique()} unique names, to {OUTPUT_FILE}" )


if __name__ == "__main__":
    main()
