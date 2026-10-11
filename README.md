# Assignment 2 — Detecting Faults on a Three-Phase Transmission Line

**Course:** AI4003 Applied Machine Learning, Fall 2026  
**Methods:** Logistic regression implemented with NumPy; linear and RBF SVMs using `sklearn.svm.SVC`.

## Project structure

```text
Assignment_2_Fault_Detection/
├── data/
│   └── detect dataset.csv
├── src/
│   ├── load_data.py
│   ├── prepare.py
│   ├── eda.py
│   ├── logistic.py
│   ├── metrics.py
│   ├── train_logistic.py
│   ├── train_svm.py
│   └── test_models.py
├── results/
│   ├── Task 1 preparation and column-inspection tables
│   ├── Three required Task 2 exploratory plots and explanations
│   ├── Two required logistic-regression cost-curve figures
│   ├── Logistic-regression weights and result table
│   ├── SVM validation table, comparison plot, and selected models
│   ├── Final test table and confusion matrices
│   ├── Task 6 analysis
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

## Dataset

The included `data/detect dataset.csv` is the dataset supplied for this project. It has 12,001 rows and nine columns. `Unnamed: 7` and `Unnamed: 8` are completely empty and are removed during preparation. The label column `Output (S)` is renamed to `fault`, where `0` means no fault and `1` means fault.

## Run instructions

Install dependencies:

```bash
pip install -r requirements.txt
```

Run from the project root, in this order:

```bash
python src/load_data.py
python src/prepare.py
python src/eda.py
python src/train_logistic.py
python src/train_svm.py
python src/test_models.py
```

Launch the frontend after training:

```bash
streamlit run app/app.py
```

## Results from the supplied dataset

- Raw rows / columns: **12,001 / 9**
- Empty columns removed: **2**
- Fault / no-fault rows: **5,496 / 6,505**
- Train / validation / test rows: **7,200 / 2,400 / 2,401**
- Chosen logistic-regression learning rates: **Set R = 1; Set Q = 10**
- Best validation SVM settings: **linear C = 0.1; RBF C = 100, gamma = 1**
- Test accuracy: **Set R logistic regression 72.64%; Set Q logistic regression 97.88%; linear SVM 73.59%; RBF SVM 99.58%**
- Test missed faults: **657, 45, 634, and 6**, respectively.

The CSV tables in `results/` contain the full metrics, confusion matrices, learned coefficients, split fractions, and validation results. The values above are generated from the included dataset.

## Assignment restrictions followed

- Data splitting uses `numpy.random.default_rng(42)` and the required 60% / 20% / remainder split.
- Feature standardisation uses training-set statistics only.
- Logistic regression, sigmoid, cost, batch gradient descent, prediction, accuracy, and confusion matrix are implemented directly using NumPy.
- The only scikit-learn model used for fitting is `sklearn.svm.SVC`.
- Validation data selects learning rates and SVM hyperparameters. The test set is used only for the final four-model evaluation.

## Communication deliverables

- `results/medium_blog_draft.md` contains the article draft based on the generated plots, results, and Task 6 analysis. Upload the required figures from `results/` into the published article.
- `results/linkedin_post_draft.md` contains the post draft. Add the published Medium and GitHub links before posting.

## Before submission

- [ ] Publish the 800–1500 word Medium article with at least two figures and the Task 6 analysis.
- [ ] Run the app and insert a real screenshot of the interface into this README, as required by the manual.
- [ ] Publish the repository publicly and make meaningful commits from both group members.
- [ ] Publish the LinkedIn post and submit its link.
