# Data preparation

## Dataset

ULB / Worldline Credit Card Fraud Detection:

- Primary source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Alternative record: https://www.openml.org/d/1597
- Machine-readable description: https://www.openml.org/api/v1/json/data/1597

Download and extract `creditcard.csv`, then place it in this directory:

```text
data/creditcard.csv
```

Do not rename an OpenML ARFF or Parquet download to CSV. Convert its format first if using that source.

## Expected schema

The original dataset contains 284,807 transactions, including 492 labeled frauds. Record and explain any changes after cleaning.

| Field | Meaning |
|---|---|
| Time | Seconds elapsed since the first transaction; not a local clock time |
| V1-V28 | Anonymized PCA-transformed numeric features |
| Amount | Transaction amount |
| Class | Target label: 0 legitimate, 1 fraud |

Each record is one transaction. The dataset does not provide reliable individual customer histories or account identifiers for constructing customer sequences.

## Initial check

From the repository root:

```bash
python code/main.py --check-data
```

Record the download date, source/version, file SHA-256 and any cleaning decisions when implementing the experiment workflow. All members should use the same raw data and split procedure.

## Sharing

Data files in this directory are ignored by Git. Keep this README in the repository and include it in the final submission if providing download instructions instead of the full dataset. Preserve source attribution and follow the dataset's applicable use terms.

