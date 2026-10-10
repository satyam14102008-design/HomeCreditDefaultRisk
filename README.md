# Home Credit Default Risk - PyCharm Project

## Dataset

Train shape: (307511, 122)
Test shape: (48744, 121)

Files:
- data/application_train.csv
- data/application_test.csv

## How to run in PyCharm

1. Open this folder in PyCharm.
2. Open the PyCharm Terminal.
3. Install packages:

```bash
pip install -r requirements.txt
```

4. Run `home_credit.py`.

Or in Terminal:

```bash
python home_credit.py
```

## Output

The program creates:

`submission.csv`

with these columns:

- `SK_ID_CURR`
- `TARGET`

## Important

`TARGET = 1` means the application has payment difficulty/default risk.
The model predicts a probability between 0 and 1.
