# product-name-classifier

Classify catalog products from the product name alone, fully offline.

[Русская версия](README.ru.md)

The original task came from a logistics project: a warehouse catalog of about 540,000 rows had to be
split into footwear and everything else. The only reliable input was a free text product name, and
the existing label column turned out to contain errors. The code here is generic, the class is
configurable, and the repository ships a synthetic catalog so it runs without any real data.

## Why the task is harder than a keyword search

A list of keywords gets surprisingly far, and then breaks on real catalog text:

- **German compound words.** `Blockabsatz-Sandaletten`, `Schlupfloafer`, `Sockenstiefel`. Each one is
  a single unseen token for a word based model.
- **Truncation.** The source system cuts every name at 40 characters, so the informative tail is often
  gone: `Friends Like These Stacked Heel Faux Sue`.
- **Several languages in one column.** German, English and Spanish names sit side by side.
- **Traps.** `Bootcut` jeans, a `U-Boot-Ausschnitt` neckline and `Sneakersocken` all contain footwear
  words and are not footwear.
- **Damaged encoding.** UTF-8 bytes decoded as cp1252 turn `Mädchen` into `MÃ¤dchen`.
- **Imbalance.** Around 7 percent of unique names belong to the positive class.

## Approach

Character n-grams instead of words, because they survive compounds, truncation and typos, and they
need no dictionary per language.

1. Repair the encoding and deduplicate names. One name repeats many times in the rows, so training on
   rows would leak the same name into both the training and the validation fold.
2. `TfidfVectorizer( analyzer="char_wb", ngram_range=( 2, 5 ), min_df=2, sublinear_tf=True )`.
3. `LogisticRegression( C=10, class_weight="balanced" )`. The class weights matter because of the
   imbalance, otherwise always answering "negative" is a good strategy for the model.
4. Honest evaluation through 5 fold stratified cross validation: every prediction comes from a model
   that has not seen that name.
5. Predictions between 0.2 and 0.8 are flagged as `needs_review` instead of being silently decided.

The training script also prints the most confident disagreements with the existing labels, sorted by
model confidence. On the real catalog this surfaced roughly 280 products that the label column called
non footwear while they clearly are footwear, which is the practical value of the whole exercise.

## Privacy

The project was built for data that must not leave the owner's machine:

- no network calls anywhere in the code, everything runs locally;
- the real catalog and the trained model are never committed. A model trained on real names stores
  tens of thousands of character fragments of those names, so `*.joblib` is in `.gitignore` next to
  `*.xlsx`;
- to verify, disconnect the machine from the internet and run it. Nothing changes.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```
uv run make_sample_data.py   # writes data/products.xlsx, a synthetic catalog
uv run train.py              # cross validation report, then saves model.joblib
uv run classify.py           # writes "data/products - classified.xlsx"
```

Useful flags:

```
uv run train.py --input path/to/catalog.xlsx --examples 10
uv run train.py --test        # first 20000 rows, does not overwrite the model
uv run classify.py --output path/to/result.xlsx
```

To run on your own file, point `--input` at it and set `NAME_COLUMN`, `LABEL_COLUMN` and, if your
label column means the opposite, `INVERT_LABEL` at the top of `train.py`.

## Output

`classify.py` copies every original column and appends three:

| Column | Meaning |
| --- | --- |
| `probability` | model confidence, 0 to 1 |
| `predicted` | 1 or 0 at a 0.5 threshold |
| `needs_review` | 1 when the probability falls between 0.2 and 0.8 |

The input file is never modified.

## Results on the bundled synthetic data

```
              precision    recall  f1-score   support
    negative      1.000     0.995     0.997       594
    positive      0.995     1.000     0.997       571
    accuracy                          0.997      1165
```

These numbers are flattering and should not be read as a quality estimate. The generator is still
simple: it produces about 1,200 unique names from a handful of templates, the classes are nearly
balanced after deduplication, and there is no label noise. Planned improvements are a Zipf
distribution of row repeats, imbalance at the unique name level, deliberately corrupted labels and
typos. On a real catalog, metrics are measured against labels that themselves contain errors, so the
disagreement list matters more than the accuracy figure.

## Files

| File | Purpose |
| --- | --- |
| `make_sample_data.py` | generates a synthetic catalog so the project runs without real data |
| `train.py` | cross validation, error report, trains and saves the model |
| `classify.py` | applies the model and writes a new Excel file |
