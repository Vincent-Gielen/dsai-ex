# DSAI Cheatsheet

## 2. Analysis: 1 Variable

### Formulas

```python
# Centrality and dispersion measures
print(f"Mean:                 {tips.tip.mean()}")
print(f"Standard deviation:   {tips.tip.std()}")     # n-1 in denominator
print(f"Variance:             {tips.tip.var()}")     # n-1 in denominator
print(f"Skewness:             {tips.tip.skew()}")
print(f"Kurtosis:             {tips.tip.kurtosis()}")

# Median & related statistics
print(f"Minimum:              {tips.tip.min()}")
print(f"Median:               {tips.tip.median()}")
print(f"Maximum:              {tips.tip.max()}")

percentiles = [0.0, 0.25, 0.5, 0.75, 1.0]
print(f"Percentiles           {percentiles}\n{tips.tip.quantile(percentiles)}")

print(f"Inter Quartile Range: {stats.iqr(tips.tip)}")
print(f"Range:                {tips.tip.max() - tips.tip.min()}")
```

## 3. Central-limit-testing

### Plots

### Formulas

## 4. Bivariate - qual + qual

### Plots

```python
# Contingency table without the margins
observed_p = pd.crosstab(rlanders.Gender, rlanders.Survey, normalize='index')
# Horizontally oriented stacked bar chart
observed_p.plot(kind='barh', stacked=True);
```

```python
# x-values:
x = np.linspace(0, 15, num=100)
# probability density of the chi-squared distribution with 4 degrees of freedom
y = stats.chi2.pdf(x, df=df)
# the number q for which the right tail probability is exactly 5%:
q = stats.chi2.isf(alpha, df=4)  # TODO: CHECK this!

fig, tplot = plt.subplots(1, 1)
tplot.plot(x, y)                     # probability density
tplot.fill_between(x, y, where=x>=q, # critical area
    color='lightblue')
tplot.axvline(q)                     # critical value
tplot.axvline(chi2, color='orange')  # chi-squared
```

### Formulas

#### 1. Cramer's V

```python
observed = pd.crosstab(rlanders.Survey, rlanders.Gender)
# observed is your contingency table (2D array or DataFrame)
cramers_v = stats.contingency.association(observed, method='cramer')
print(f"Cramer's V: {cramers_v}")
```

#### 2. Chi-squared test for independence

```python
# Chi-squared test for independence based on a contingency table
observed = pd.crosstab(rlanders.Survey, rlanders.Gender)
chi2, p, df, expected = stats.chi2_contingency(observed)

print("Chi-squared       : %.4f" % chi2)
print("Degrees of freedom: %d" % df)
print("P-value           : %.4f" % p)

# Calculate critical value
alpha = .05
g = stats.chi2.isf(alpha, df = df)
print("Critical value     : %.4f" % g)
```

#### 3. Goodness-of-fit test

```python
observed =   np.array([   127,      75,      98,     27,     73])
expected_p = np.array([   .35,     .17,     .23,    .08,    .17])
alpha = 0.05               # Significance level
n = sum(observed)          # Sample size
k = len(observed)          # Number of categories
dof = k - 1                # Degrees of freedom
expected = expected_p * n  # Expected absolute frequencies in the sample
g = stats.chi2.isf(alpha, df=dof)  # Critical value

# Goodness-of-fit-test in Python:
chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)

print("Significance level  ⍺ = %.2f" % alpha)
print("Sample size         n = %d" % n)
print("k = %d; df = %d" % (k, dof))
print("Chi-squared        χ² = %.4f" % chi2)
print("Critical value      g = %.4f" % g)
print("p-value             p = %.4f" % p)
```

#### 4. Standardised residuals

```python
# Data frame with 2 columns:
#  - number of boys in the family (index)
#  - number of families in the sample with that number of boys
families = pd.DataFrame(
    np.array(
        [[0,  58],
         [1, 149],
         [2, 305],
         [3, 303],
         [4, 162],
         [5,  45]]),
    columns=['num_boys', "observed"])
families.set_index(['num_boys'])
n = families.observed.sum() # sample size

from scipy.special import binom # binomial-function

# probability for a boy
prob_boy = .5
# Add new colum to the data frame for the expected percentages
families['expected_p'] = binom(5, families.num_boys) * prob_boy**families.num_boys * prob_boy**(5-families.num_boys)
# Expected absolute frequencies in the sample:
families['expected'] = families['expected_p'] * n
print(families)

alpha=0.01                         # significance level
dof=len(families)-1                # degrees of freedom
g = stats.chi2.isf(alpha, df=dof)  # Critical value
# Perform Chi-squared test, calculate χ² and p
chi2, p = stats.chisquare(f_obs=families.observed, f_exp=families.expected)

print("Chi-squared   χ² = %.4f" % chi2)
print("Critical value g = %.4f" % g)
print("p-value        p = %f"   % p)

(observed - expected) / np.sqrt(expected * (1-expected_p))
# Dit moet binnen [-2, 2] liggen.
```

## 5. Bivariate - qual + quant

### Plots

```python
# Sample:
control = np.array([91, 87, 99, 77, 88, 91])
treatment = np.array([101, 110, 103, 93, 99, 104])
# Visualization:
sns.boxplot(
    data=pd.DataFrame({'control': control, 'treatment': treatment}),
    orient='h');
```

### Formulas

#### The t-test for two independent samples

```python

# Sample:
control = np.array([91, 87, 99, 77, 88, 91])
treatment = np.array([101, 110, 103, 93, 99, 104])

stats.ttest_ind(a=control, b=treatment,
    alternative='less', equal_var=False)
#we verwachten hier dat control less is dan treatment, andere opties in alternative zijn mogelijk
```

#### The t-test for paired samples

```python
# Paired t-test with ttest_rel()
stats.ttest_rel(regular, additives, alternative='less')
```

#### Effect size - Cohen's d

```python
#Effect size is another metric to express the magnitude of the difference between two groups. Several definitions of effect size exist, but one of the most commonly used is Cohen's d.

#Cohen's d is defined as the difference between the means of both groups, divided by a pooled standard deviation. There's no Python function for calculating Cohen's d readily available, so we define it here, according to the formula:

def cohen_d(a, b):
    na = len(a)
    nb = len(b)
    pooled_sd = np.sqrt( ((na-1) * np.var(a, ddof=1) +
                          (nb-1) * np.var(b, ddof=1)) / (na + nb - 2) )
    return (np.mean(b) - np.mean(a)) / pooled_sd

# Effect size of additives in gasoline:
cohen_d(regular, additives)

```
