# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

## Objective
Build ensemble models (Decision Trees, Random Forest, Gradient Boosting),
tune them systematically with GridSearchCV, evaluate generalization via
cross-validation and a manual learning curve, and package the best model
into a reproducible, serialized sklearn Pipeline.

## Dataset
`cleaned_data.csv` (from Part 1), preprocessed identically to Part 2 —
2,930 rows × 255 encoded features, classification target `y_clf`
(above/below median `SalePrice`).

## Tasks Performed
- Trained an unconstrained Decision Tree baseline (demonstrates overfitting)
- Trained a controlled Decision Tree (`max_depth=5`, `min_samples_split=20`)
- Compared Gini vs. Entropy split criteria
- Trained Random Forest (`n_estimators=100`, `max_depth=10`), examined
  feature importances
- Trained Gradient Boosting (`n_estimators=100`, `learning_rate=0.1`,
  `max_depth=3`)
- Ran a feature ablation study, removing the 5 lowest-importance features
- Ran 5-fold cross-validation across Logistic Regression, Decision Tree,
  Random Forest, and Gradient Boosting
- Ran GridSearchCV over an 18-configuration Random Forest parameter grid
  inside a `Pipeline` (imputer + scaler + classifier)
- Built a manual learning curve (5 training-set fractions)
- Serialized the best pipeline with `joblib`, confirmed reload-and-predict
- Compiled a final summary comparison table and model recommendation

## How to Run
```bash
pip install pandas numpy scikit-learn joblib
```
Requires `cleaned_data.csv` from Part 1. Run all cells top to bottom in
Jupyter or Google Colab (preprocessing cells from Part 2 are repeated at
the top of this notebook so it runs standalone).

## Key Findings & Design Decisions

### Task 1 — Decision Tree baseline (overfitting)
Training accuracy = **100.00%**, Test accuracy = **90.96%**, gap = **9.04
points** — clear overfitting. Decision trees are **high-variance models**
because they build their structure greedily: at each node, the tree picks
the single best split for the data it currently sees, without ever
revisiting or reconsidering earlier splits. With no depth limit, the tree
splits until it perfectly separates the training data, memorizing noise
rather than learning generalizable patterns.

### Task 2 — Controlled Decision Tree
With `max_depth=5` and `min_samples_split=20`: training accuracy dropped to
**92.75%**, test accuracy **improved slightly** to **91.64%**, and the gap
shrank to just **1.11 points** (from 9.04). **`max_depth`** limits how deep
the tree can grow, reducing variance at the cost of some bias. **
`min_samples_split=20`** prevents splitting a node with fewer than 20
samples, avoiding splits driven by noise in small subsets. Together these
constraints turned an overfit tree into a well-generalizing one.

### Task 3 — Gini vs. Entropy
- **Gini impurity** = 1 − Σ pᵢ²
- **Entropy** = −Σ pᵢ log₂(pᵢ)

**Gini = 0** means the node is perfectly pure — every sample belongs to the
same class. Results (max_depth=5): Gini test accuracy = **92.32%**, Entropy
= **91.64%** — nearly identical, since both criteria measure node impurity
in mathematically related ways.

### Task 4 — Random Forest
`n_estimators=100, max_depth=10`: Training accuracy = **98.68%**, Test
accuracy = **94.03%**, ROC-AUC = **0.9908**. The 4.65-point train-test gap
is smaller than the single unconstrained tree's 9.04-point gap, showing
ensemble averaging reduces overfitting even with a fairly deep per-tree
limit.

**Top 5 features by importance:** `Gr Liv Area` (0.0825), `Full Bath`
(0.0771), `Year Built` (0.0572), `Overall Qual` (0.0551), `Garage Area`
(0.0474).

**How Random Forest computes feature importance:** the average reduction in
Gini impurity contributed by that feature, across every split, across every
tree in the forest — a feature that consistently helps separate classes
well gets a high score. This differs from a linear regression coefficient,
which represents a fixed, linear, additive, *directional* effect (holding
other features constant); importance captures overall usefulness for
correct splits across potentially non-linear, interaction-heavy trees, but
doesn't indicate direction the way a coefficient does.

**Bagging concept:** Random Forest builds each tree using **bootstrap
sampling** — every tree sees a random sample drawn *with replacement* from
the training data, so each tree trains on a slightly different subset. At
each split, only a random subset of **√(number of features)** is
considered, forcing trees to diversify further. Because each tree makes
different, partially-uncorrelated errors, averaging (or majority-voting)
across many diverse trees cancels out much of this individual variance,
producing a far more stable, generalizable prediction than any single deep
tree.

### Task 4a — Gradient Boosting
`n_estimators=100, learning_rate=0.1, max_depth=3`: Training accuracy =
**97.61%**, Test accuracy = **94.88%** (highest test accuracy of any single
model tried), ROC-AUC = **0.9904**. Unlike Random Forest's parallel,
independent trees, Gradient Boosting builds trees **sequentially**, each
new tree correcting the errors of the previous ones — often yielding
slightly better accuracy.

### Task 4b — Feature ablation study
The 5 lowest-importance features were rare one-hot dummy categories:
`Sale Type_ConLw`, `Misc Feature_TenC`, `Sale Type_Con`, `Sale Type_ConLI`,
`MS SubClass_40`. Full-model test AUC = **0.9908**; reduced model (5
features removed) test AUC = **0.9897** — a difference of just **0.0011**.

This tiny gap indicates these features were **genuinely uninformative**
rather than meaningfully contributing — likely rare categories with too few
observations to provide reliable signal. Practically, this means a
**simpler, lower-dimensional model** (250 vs. 255 features) would have
lower inference cost and reduced maintenance burden, and here that
simplification comes at an acceptably small AUC cost — well within a
reasonable production tolerance threshold.

### Task 5 — Cross-validated comparison
| Model | Mean AUC | Std AUC |
|---|---|---|
| Logistic Regression | 0.9685 | 0.0046 |
| Decision Tree (max_depth=5) | 0.9279 | 0.0035 |
| Random Forest | 0.9794 | 0.0044 |
| Gradient Boosting | **0.9812** | 0.0032 |

A single train-test split gives one performance estimate that depends
heavily on which rows happened to land in the test set. 5-fold
cross-validation trains and evaluates each model 5 separate times, using a
different fold as the held-out set each time, giving a far more reliable
average estimate of generalization performance — plus a standard deviation
showing how consistent that performance is across different data subsets.
Gradient Boosting has both the highest mean AUC and tightest std here.

### Task 6 — GridSearchCV
**Best params:** `max_depth=None, min_samples_leaf=1, n_estimators=200`
**Best score (mean CV AUC): 0.9803**

The grid has 3 × 3 × 2 = **18 total configurations**, each evaluated across
5 folds = **90 total model fits**. Exhaustive Grid Search tries every
combination, guaranteeing the best result *within the specified grid*, but
its cost grows multiplicatively with each added parameter/value — with many
hyperparameters, this becomes computationally prohibitive. Randomized
Search instead samples a fixed number of random combinations from the grid
(or from continuous distributions), trading a small chance of missing the
absolute best combination for dramatically lower computational cost,
especially valuable when some hyperparameters matter far more than others
and exhaustive coverage of all of them isn't necessary.

### Task 7 — Manual learning curve
| Training fraction | Training AUC | Test AUC |
|---|---|---|
| 0.2 | 1.0000 | 0.9891 |
| 0.4 | 1.0000 | 0.9898 |
| 0.6 | 1.0000 | 0.9891 |
| 0.8 | 1.0000 | 0.9902 |
| 1.0 | 1.0000 | 0.9915 |

**(i)** Training AUC does **not** decrease as the training set grows — it
stays at a perfect 1.0000 throughout. This is because the GridSearchCV-
selected best model has `max_depth=None`, giving individual trees more than
enough capacity to perfectly fit any subset size tested here (even the
smallest, ~469 rows).

**(ii)** Test AUC does increase with more training data overall — from
0.9891 at 20% to 0.9915 at 100% — though not perfectly monotonically (a
small dip at 60%). This suggests more training data would likely help
further.

**(iii)** Conclusion: the model appears **more data-limited than
capacity-limited** — test AUC is still rising at 100% and hasn't clearly
plateaued, while training AUC being pinned at the ceiling (rather than
gradually decreasing, as would happen if the model ran out of capacity)
shows it has more capacity than the current data size requires. Collecting
more training data would likely improve test performance further.

### Task 8 — Serialized model
`best_model.pkl` saved via `joblib.dump()`. Reload-and-predict block ran
successfully: predictions `[0, 0]`, probabilities `[0.325, 0.015]` — no
errors.

### Task 9 — Summary comparison table
| Model | 5-fold CV Mean AUC | 5-fold CV Std AUC | Test-set AUC |
|---|---|---|---|
| Logistic Regression | 0.9685 | 0.0046 | 0.9820 |
| Decision Tree (max_depth=5) | 0.9279 | 0.0035 | 0.9604 |
| Random Forest | 0.9794 | 0.0044 | 0.9908 |
| Gradient Boosting | 0.9812 | 0.0032 | 0.9904 |
| **Tuned RF (GridSearchCV)** | **0.9803** | — | **0.9915** |

*(Note: Part 2's additional variants — Logistic Regression C=0.01 [test
AUC=0.9905] and the Part 2 bonus Random Forest [test AUC=0.9915] — are not
included above since they weren't evaluated with 5-fold CV in this
notebook; they're noted here for context.)*

**Recommendation:** I would recommend the **Tuned Random Forest
(GridSearchCV)** to the client. It achieves the highest test-set AUC
(0.9915) of any model evaluated, and its cross-validated mean AUC (0.9803)
is competitive with Gradient Boosting's (0.9812) while being derived from a
systematic, exhaustive search over 18 configurations rather than a single
manually-chosen setting. It also comes pre-packaged as a complete
`Pipeline` (imputation + scaling + model), making it directly deployable
without needing to separately manage preprocessing steps — a meaningful
practical advantage for production use. While Gradient Boosting has a
marginally higher CV mean, the tuned Random Forest's combination of top
test performance, deployment-readiness, and robustness (ensemble averaging
across 200 trees) makes it the safer, more production-ready choice.

## Output Files
- `best_model.pkl` — serialized best Pipeline from GridSearchCV

## Libraries Used
- Pandas, NumPy
- scikit-learn (`DecisionTreeClassifier`, `RandomForestClassifier`,
  `GradientBoostingClassifier`, `LogisticRegression`, `Pipeline`,
  `SimpleImputer`, `StandardScaler`, `GridSearchCV`, `StratifiedKFold`,
  `cross_val_score`, metrics)
- joblib
