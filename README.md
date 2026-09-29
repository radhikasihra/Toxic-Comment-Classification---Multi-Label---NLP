# Toxic Comment Classification | Multi-Label NLP

Classify Wikipedia comments across six types of harmful language. A comment can receive more than one label, making this a **multi-label text classification** project.

## Project at a glance

| Area | Details |
| --- | --- |
| Problem | Identify toxic, severe toxic, obscene, threat, insult, and identity hate comments |
| Data | 159,571 human-labeled Wikipedia comments from the Jigsaw toxic comment dataset |
| Approach | Text cleaning and stemming → TF-IDF features → one-vs-rest classifiers |
| Models | Multinomial Naive Bayes and Logistic Regression |
| Best recorded result | Logistic Regression: **0.9795 ROC-AUC** and **0.9187 exact-match accuracy** on a 20% holdout split |
| Tools | Python, pandas, NumPy, NLTK, scikit-learn, Matplotlib, Seaborn |

## Why this matters

Online comments can contain several kinds of abuse at once. This notebook explores label frequency and overlap, prepares comment text, compares two classical NLP baselines, and evaluates predictions for each label. It illustrates a practical moderation use case and the challenges of imbalanced labels.

## Workflow

1. Load `train.csv` and inspect missing values and descriptive statistics.
2. Plot label counts and the number of labels per comment.
3. Remove the ID column; remove stopwords, normalize text, and apply Snowball stemming.
4. Split comments into training and test sets (80/20, `random_state=42`).
5. Fit TF-IDF and one-vs-rest Naive Bayes and Logistic Regression pipelines.
6. Report ROC-AUC, exact-match accuracy, and per-label precision, recall, and F1; test sample comments and plot ROC curves.

## Results recorded in the notebook

| Model | ROC-AUC | Exact-match accuracy | Micro F1 |
| --- | ---: | ---: | ---: |
| Multinomial Naive Bayes | 0.8604 | 0.8998 | 0.22 |
| Logistic Regression | **0.9795** | **0.9187** | **0.68** |

The ROC-AUC values are the notebook's default multi-label average. **Exact-match accuracy** requires every predicted label for a comment to be correct. The dataset is highly imbalanced: 143,346 of 159,571 comments have no positive toxicity label. Accuracy alone therefore does not describe performance well. In the Logistic Regression report, recall is 0.15 for `threat` and 0.16 for `identity_hate`, so these categories need further work before relying on the model for moderation decisions.

## Run the notebook

1. Download the Jigsaw **Toxic Comment Classification Challenge** training data from Kaggle and place `train.csv` next to the notebook. The expected columns are `id`, `comment_text`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.
2. Install the dependencies:

   ```bash
   pip install notebook pandas numpy nltk scikit-learn matplotlib seaborn
   ```

3. Download the NLTK English stopword corpus once:

   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

4. Open `Toxic Comment Classification - Multi Label - NLP.ipynb` in Jupyter and run its cells in order.

The notebook expects the local training CSV; it does not download the dataset or save a trained model. The final ROC plot cell refits the Logistic Regression pipeline separately for each label, so rerun the earlier Logistic Regression training cell before reusing that pipeline for six-label predictions.

## Possible improvements

- Tune decision thresholds for each label and report precision-recall curves, especially for rare labels.
- Compare stratified multi-label splits or cross-validation for more stable estimates.
- Save the fitted pipeline and expose inference through a small app or API.

**Repository contents:** this README and the Jupyter notebook. Add the dataset only if its distribution terms allow it.
