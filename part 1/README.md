# Part 1 — Exploratory Data Analysis (EDA)

## Objective
Perform exploratory data analysis on the Ames Housing dataset to understand
data distribution, missing-value patterns, relationships between features,
and the variables most likely to affect house sale prices — producing a
clean dataset ready for Part 2's modeling.

## Dataset
`AmesHousing.csv` 

## Tasks Performed
- Loaded dataset using Pandas, inspected shape/dtypes/head
- Computed null count and percentage for every column; flagged 6 columns
  exceeding 20% nulls; median-imputed 38 numeric columns below 20% nulls
- Checked and confirmed zero duplicate records
- Corrected data types: `MS SubClass` (int → category, since it's a building
  code, not a true quantity) and `Neighborhood` (object → category)
- Generated summary statistics (`df.describe()`) and computed skewness for
  every numeric column
- Detected outliers in `SalePrice` and `Gr Liv Area` using the IQR method
  (documented, not dropped)
- Created all 5 required visualizations plus a correlation heatmap
- Compared mean vs. median imputation strategy for the two most-skewed columns
- Computed Spearman rank correlation and compared against Pearson
- Performed grouped aggregation (`SalePrice` by `Overall Qual`)
- Saved the cleaned dataset to `cleaned_data.csv`

## How to Run
```bash
pip install pandas numpy matplotlib seaborn
```
Open in Jupyter or Google Colab, upload `AmesHousing.csv` when prompted, and
run all cells top to bottom.

## Key Findings

### Null value analysis
Six columns exceed 20% nulls and were left un-imputed: `Pool QC` (99.56%),
`Misc Feature` (96.38%), `Alley` (93.24%), `Fence` (80.48%), `Mas Vnr Type`
(60.58%), `Fireplace Qu`. Most of these represent features that simply don't
exist for most houses (a NaN in `Pool QC` almost always means "no pool," not
a data gap). For the 38 numeric columns below 20% nulls, missing values were
filled with the **median rather than the mean**, since several numeric
columns are right-skewed with a handful of large/luxury properties pulling
the mean upward — the median better represents a typical house.

### Duplicates
No exact-duplicate rows were found (`df.duplicated().sum() = 0`); no rows
removed, null percentages unaffected.

### Data type correction
`MS SubClass` was stored as `int64` (e.g. 20, 60, 120) despite being a
categorical building-type code, not a real quantity — converted to
`category`. `Neighborhood` (28 unique values) was also converted to
`category`, reducing memory from 7,062.66 KB to 6,909.29 KB (153.37 KB / 2.2%
saved). Total dataset memory dropped from 7,081.35 KB to 6,909.29 KB overall.

### Skewness
`Misc Val` is the most skewed column (**skew = 22.000**), followed by
`Pool Area` (16.939) — both **positively skewed**: most houses have 0 (no
miscellaneous features / no pool), with rare outliers pulling the mean well
above the median. **Consequence:** mean-imputation here would inject an
unrealistic non-zero value into rows that most likely have none of that
feature — the median (0 for both) is the appropriate statistic.

### IQR outliers
- `SalePrice`: bounds [3,500, 339,500] → **137 rows (4.68%)** outside.
- `Gr Liv Area`: bounds [200.88, 2,667.88] → **75 rows (2.56%)** outside.

Retained (not dropped) in `cleaned_data.csv` — these almost certainly
represent genuine large/luxury properties. For Part 2, the plan is to **cap
(winsorize)** both at their IQR bounds to limit their influence on the
regression model while preserving every row.

### Imputation strategy comparison (Task 8a)
| Column | Mean (pre-impute) | Median (pre-impute) | Skew |
|---|---|---|---|
| `Misc Val` | 50.635 | 0.000 | 22.000 |
| `Pool Area` | 2.243 | 0.000 | 16.939 |

Both positively skewed; median (0) is far more representative than the mean,
so median was used. Both columns already had 0 nulls at this point (imputed
earlier in the <20%-null step) — `isnull().sum()` confirms 0 remain.

### Spearman vs. Pearson (Task 8b)
| Pair | Pearson | Spearman | \|Diff\| |
|---|---|---|---|
| `Lot Frontage` × `Lot Area` | 0.363 | 0.573 | 0.210 |
| `Open Porch SF` × `Year Built` | 0.198 | 0.403 | 0.205 |
| `Garage Yr Blt` × `Open Porch SF` | 0.220 | 0.394 | 0.173 |

All three show **|Spearman| > |Pearson|**, indicating **monotonic but
non-linear** relationships — the variables move together consistently but
not proportionally. For Part 2 feature-selection guidance, I will rely on
**Spearman**, since it captures these relationships more accurately; Pearson
understates how related these pairs actually are.

### Grouped aggregation (Task 8c)
`SalePrice` by `Overall Qual`:

| Overall Qual | Mean | Std | Count |
|---|---|---|---|
| 10 | 450,217.32 | 141,975.97 | 31 |
| 9 | 368,336.77 | 79,201.27 | 107 |
| 8 | 270,913.59 | 61,326.21 | 350 |
| 7 | 205,025.76 | 43,166.27 | 602 |
| 6 | 162,130.32 | 37,201.30 | 732 |
| 5 | 134,752.52 | 27,690.60 | 825 |
| 4 | 106,485.10 | 29,224.94 | 226 |
| 3 | 83,185.98 | 23,569.80 | 40 |
| 2 | 52,325.31 | 17,562.96 | 13 |
| 1 | 48,725.00 | 29,341.94 | 4 |

**(a)** Highest mean and highest std dev: both **Quality 10** ($450,217.32
mean, $141,975.97 std — the latter likely inflated by a small n=31 with a
few very-high-end outliers). **(b)** Within-group std is mostly modest
relative to the mean (e.g. Quality 5: ~21%) — a tight, informative spread,
signaling `Overall Qual` is a reliable predictive feature rather than one
where noise swamps the signal. **(c)** Ratio of highest to lowest group mean
= **9.240** — a very large gap, strongly confirming `Overall Qual` carries
substantial predictive signal, consistent with its 0.80 Pearson correlation
with `SalePrice`.

## Visualizations
1. **Sale Price Across Houses by Row Index (Line Plot)** — `SalePrice`
   plotted by row index shows no discernible trend, confirming row order
   carries no inherent structure (not sorted by price or time).
2. **Mean Sale Price by Neighborhood (Bar Chart)** — substantial variation
   across 28 neighborhoods, from ~$95K (MeadowV) to ~$330K (NoRidge) —
   neighborhood location is clearly associated with price level.
3. **Distribution of Misc Val (Histogram)** — extremely right-skewed
   (skew=22.00); ~2,830 of 2,930 houses have `Misc Val = 0`, with a small
   tail scattered up to $1,000+ (view clipped at the 99th percentile).
4. **Gr Liv Area vs SalePrice (Scatter Plot)** — strong positive correlation
   (r = 0.707); living area explains roughly 50% of the variance in price.
5. **Sale Price by Overall Quality Rating (Box Plot)** — medians rise
   steadily and substantially from quality 1 through 10, with spread also
   widening at higher quality levels.
6. **Correlation Heatmap** — 12 key numeric columns with `annot=True`.
   Highest genuine pairwise correlation (excluding row-identifier columns
   `Order`/`PID`, which produce spurious artifacts — e.g. `Order` vs `Yr Sold`
   showed r=-0.976 purely from dataset compilation order) is
   **`Garage Area` & `Garage Cars` (r = 0.890)** — very likely a direct,
   near-mechanical relationship (a garage's stated car capacity is largely
   determined by its physical square footage) rather than one requiring a
   third-variable explanation, though overall house/lot size could
   indirectly drive both simultaneously.

## Output Files
- `plot1_line.png`
- `plot2_bar.png`
- `plot3_histogram.png`
- `plot4_scatter.png`
- `plot5_boxplot.png`
- `plot6_heatmap.png`
- `cleaned_data.csv`

## Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
