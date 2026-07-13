# Part 2 — Supervised Machine Learning Model: Build, Train, and Evaluate

## Objective
Build and rigorously evaluate two predictive models on the Ames Housing
dataset: a regression model predicting `SalePrice`, and a binary
classification model predicting whether a house sells above or below the
median price — with leak-free preprocessing and imbalance handling where
needed.

## Dataset
`cleaned_data.csv` (from Part 1) — 2,930 rows × 82 columns of Ames, Iowa
house sale records.

## Tasks Performed
- Loaded `cleaned_data.csv`, capped `SalePrice` and `Gr Liv Area` at their
  Part-1-identified IQR bounds
- Defined `y_reg` (SalePrice), `y_clf` (median-split binary label), and `X`
  (79 features, with row-identifier columns `Order`/`PID` removed)
- Encoded 10 ordinal quality/condition columns via label encoding and 33
  nominal columns via one-hot encoding (`drop_first=True`)
- Performed a leak-free 80/20 train-test split and fit `StandardScaler` on
  training data only
- Trained Linear Regression and Ridge Regression (alpha=1.0), compared MSE/R²
- Checked class balance, trained Logistic Regression, evaluated with
  confusion matrix, classification report, ROC curve, and AUC
- Ran decision-threshold sensitivity analysis (0.30–0.70)
- Compared strongly-regularized (C=0.01) vs. baseline (C=1.0) logistic
  regression
- Ran a 500-sample bootstrap to quantify the reliability of the AUC
  difference between the two regularization strengths
- **Bonus:** trained Random Forest regressor/classifier for comparison
  against the required linear models

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib
```
Requires `cleaned_data.csv` from Part 1 in the same folder. Run all cells
top to bottom in Jupyter or Google Colab.

## Key Findings & Design Decisions

### Target label definitions
- **`y_reg`** (regression target) = `SalePrice`, IQR-capped (max reduced from
  $755,000 to $339,500) per Part 1's outlier-handling plan.
- **`y_clf`** (classification target) = binarized `y_reg` at its median
  ($160,000): `1` = above-median-priced house (49.93%), `0` = at/below-median
  (50.07%) — a near-perfectly balanced split by construction.

### Categorical encoding
- **Label encoding** (10 ordinal columns): `Exter Qual`, `Exter Cond`,
  `Bsmt Qual`, `Bsmt Cond`, `Heating QC`, `Kitchen Qual`, `Fireplace Qu`,
  `Garage Qual`, `Garage Cond`, `Pool QC` — all mapped Poor(1) < Fair(2) 
  Typical/Average(3) < Good(4) < Excellent(5), with "None" (feature absent,
  e.g. no basement) mapped to 0. This preserves the genuine quality ranking
  these columns represent.
- **One-hot encoding** (33 nominal columns, `drop_first=True`):
  `Neighborhood`, `Roof Style`, `Exterior 1st`, `Sale Type`, etc. — these
  have no inherent order; label-encoding them would force a false numeric
  ranking (e.g. implying one neighborhood is "greater than" another), which
  one-hot encoding avoids by giving each category its own independent 0/1
  column.
- Columns where `NaN` means "feature doesn't exist" (`Pool QC`, `Alley`,
  `Fence`, `Misc Feature`, `Fireplace Qu`, `Mas Vnr Type`, garage/basement
  quality columns) were filled with the string `"None"` as their own
  category before encoding, rather than treated as missing data.
- Final encoded feature matrix: **79 → 255 columns**.

### Leak-free split and scaling
Data was split 80/20 (`X_train`: 2,344 rows, `X_test`: 586 rows) with
`random_state=42`. `StandardScaler` was fit **only on `X_train`**
(`scaler.fit(X_train)`), then applied to both sets via `.transform()`.
**Fitting the scaler on the full dataset would constitute data leakage**,
because the scaler's mean/std would then be computed using information from
the test set — rows the model is supposed to have never seen. This would
let test-set statistics subtly influence the "learned" scale applied during
training, making evaluation metrics look artificially better than the
model's true real-world performance.

### Regression — Linear Regression & Ridge
| Model | MSE | R² |
|---|---|---|
| Linear Regression | 548,293,407.64 | 0.8971 |
| Ridge (alpha=1.0) | 471,428,807.99 | 0.9115 |

Linear Regression explains ~90% of the variance in `SalePrice`.

**Top 3 coefficients by magnitude:** `BsmtFin SF 1` (+95,288.83),
`Bsmt Unf SF` (+85,233.63), `Total Bsmt SF` (-79,960.15). A **large positive
coefficient** means a one-unit increase in that (scaled) feature is
associated with an increase in predicted `SalePrice` by roughly that many
dollars, holding other features constant. A **large negative coefficient**
means the opposite — `Total Bsmt SF`'s negative coefficient here reflects
multicollinearity with its component parts (`BsmtFin SF 1`+`BsmtFin SF 2`+
`Bsmt Unf SF` sum to `Total Bsmt SF`), not a genuine "more basement area
lowers price" effect.

**Ridge vs. Linear:** Ridge adds an L2 penalty (`alpha × Σcoefficient²`) to
the loss function, shrinking large coefficients toward zero to reduce
overfitting. `alpha` controls the strength of this penalty. With 255
features and only 2,344 training rows, plain OLS is prone to unstable
coefficients; Ridge's penalty improved generalization here, raising R² from
0.8971 to 0.9115 and lowering MSE by ~14%.

### Classification — Logistic Regression
**Class balance check:** `y_clf_train` minority class = 49.40%
(`{0: 1186, 1: 1158}`) — above the 35% imbalance threshold, so **no
imbalance-handling technique (SMOTE / class_weight) was applied**.

**Before/after comparison:** Since no resampling was needed, the "before"
and "after" class counts for `y_clf_train` are identical:
`{0: 1186 (50.60%), 1: 1158 (49.40%)}`. This itself is the required
before/after demonstration — it shows the imbalance check was explicitly
performed and confirms no correction was necessary, since the classes are
already near-balanced by construction (median split).

**Results (C=1.0, threshold=0.5):**
- Confusion Matrix: `[[267, 14], [23, 282]]`
- Accuracy: 0.9369
- Precision/Recall/F1 (class 0): 0.92 / 0.95 / 0.94
- Precision/Recall/F1 (class 1): 0.95 / 0.92 / 0.94
- **AUC = 0.9819**

**(a) Formulas:** Precision = TP/(TP+FP), Recall = TP/(TP+FN).
**(b) Which matters more:** For this house-price-tier task, misclassifying
carries no strongly asymmetric real-world cost — precision and recall are
roughly equally important, and the model achieves high, balanced values for
both.
**(c) AUC interpretation:** AUC = 0.9819 means the model correctly ranks a
random above-median house higher than a random below-median house 98.19% of
the time — excellent discriminative ability.

### Decision-threshold sensitivity (Task 5b)
| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.9074 | 0.9639 | 0.9348 |
| 0.40 | 0.9263 | 0.9475 | 0.9368 |
| 0.50 | 0.9527 | 0.9246 | 0.9384 |
| 0.60 | 0.9590 | 0.9213 | 0.9398 |
| 0.70 | 0.9685 | 0.9082 | 0.9374 |

**(a)** Formulas as above. **(b)** F1-maximizing threshold: **0.60**
(F1=0.9398). **(c)** Precision/recall roughly equally important (as above).
**(d)** I would raise the threshold slightly to 0.60 rather than the default
0.50 — trading a small recall drop (0.9246→0.9213) for a larger precision
gain (0.9527→0.9590), a net F1 improvement. Raising further (e.g. to 0.70)
costs more recall for diminishing precision returns.

### Regularization experiment (Task 6)
| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| C=1.0 (baseline) | 0.9369 | 0.9527 | 0.9246 | 0.9820 |
| C=0.01 (strong regularization) | 0.9454 | 0.9596 | 0.9344 | 0.9905 |

`C` is the **inverse** of regularization strength — smaller `C` means a
stronger L2 penalty. Reducing `C` to 0.01 **improved** performance across
all four metrics, likely because with 255 features and only 2,344 training
rows, the C=1.0 baseline was mildly overfitting.

### Bootstrap confidence interval for AUC difference (Task 7)
Using 500 bootstrap samples:
- **Mean AUC difference (C=1.0 − C=0.01): -0.0087**
- **95% CI: [-0.0148, -0.0034]**

The interval **excludes zero**, meaning C=0.01's advantage over C=1.0 is
statistically reliable across resampled test sets, not a one-split
coincidence.

## Bonus: Random Forest (Additional Model Comparison)
As an extension beyond the required models, a Random Forest was trained on
the same scaled train/test split for direct comparison.

### Regression comparison
| Model | MSE | R² |
|---|---|---|
| Linear Regression | 548,293,407.64 | 0.8971 |
| Ridge (alpha=1.0) | 471,428,807.99 | 0.9115 |
| Random Forest | 371,478,880.73 | **0.9303** |

Random Forest outperforms both linear models, explaining ~93% of the
variance in `SalePrice`. Its top features by importance — `Overall Qual`
(0.596), `Gr Liv Area` (0.112), `Garage Area` (0.035) — align closely with
the strongest linear correlations identified in Part 1 (`Overall Qual`:
r=0.80, `Gr Liv Area`: r=0.71), but Random Forest captures additional
non-linear/interaction effects that linear models can't, explaining its
higher R². See `plot8_rf_feature_importance.png` and
`plot9_regression_comparison.png`.

### Classification comparison
| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression (C=1.0) | 0.9369 | 0.9820 |
| Logistic Regression (C=0.01) | 0.9454 | 0.9905 |
| Random Forest | **0.9471** | **0.9915** |

Random Forest also edges out both logistic regression variants on accuracy
and AUC, confirming that some of the remaining signal in this dataset is
non-linear — tree-based models can naturally capture feature interactions
(e.g. how quality and size jointly affect price) that a linear decision
boundary cannot. See `plot10_classification_comparison.png`.

**Takeaway:** while the required linear models already performed strongly,
Random Forest demonstrates that a small additional performance gain is
available by modeling non-linear relationships — useful context for anyone
choosing a model for production deployment where the highest possible
accuracy matters more than interpretability.

## Conclusion
Both required models performed strongly on this dataset. Linear/Ridge
Regression explained ~90-91% of the variance in `SalePrice`, with Ridge's
regularization providing a genuine, measurable improvement given the high
feature-to-sample ratio (255 features, 2,344 training rows). The
classification model achieved 93.7% accuracy and 0.982 AUC, confirming that
house-price tier is highly predictable from structural/quality features
like basement area, living area, and overall quality. The regularization
experiment showed that a stronger L2 penalty (C=0.01) reliably outperforms
the weaker baseline — a bootstrap 95% CI that excludes zero confirms this
isn't due to chance. The bonus Random Forest models further improved on
every metric (R²=0.930, Accuracy=0.947, AUC=0.992), showing that some
non-linear signal remains beyond what linear models can capture. Overall,
the results validate Part 1's EDA finding that this dataset's numeric
features carry strong, genuine predictive signal for house prices, and
demonstrate a complete leak-free ML pipeline from raw data through multiple
rigorously evaluated regression and classification models.

## Output Files
- `plot7_roc_curve.png`
- `plot8_rf_feature_importance.png`
- `plot9_regression_comparison.png`
- `plot10_classification_comparison.png`

## Libraries Used
- Pandas
- NumPy
- scikit-learn (`LinearRegression`, `Ridge`, `LogisticRegression`,
  `RandomForestRegressor`, `RandomForestClassifier`, `StandardScaler`,
  `train_test_split`, metrics)
- Matplotlib
