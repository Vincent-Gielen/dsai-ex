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
# We denken: 2 verschillende testgroepen
# Sample:
control = np.array([91, 87, 99, 77, 88, 91])
treatment = np.array([101, 110, 103, 93, 99, 104])

stats.ttest_ind(a=control, b=treatment,
    alternative='less', equal_var=False)
#we verwachten hier dat control less is dan treatment, andere opties in alternative zijn mogelijk
```

#### The t-test for paired samples

```python
# We denken: 1 groep, met verschillende omstandigheden
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

## 6. Bivariate - quant + quant -  Regression analysis

### Plots

```python
penguins = sns.load_dataset('penguins')
# Independent variable on X-axis, dependent on Y-axis
sns.relplot(data=penguins,
            x='flipper_length_mm', y='body_mass_g', 
            hue='species', style='sex');
```

```python
sns.scatterplot(data=male_chinstrap, x='flipper_length_mm', y='body_mass_g');
```

```python
# Plot a data set with its regression line
sns.lmplot(data=male_chinstrap, x='flipper_length_mm', y='body_mass_g');
```

### Formulas

#### Method of least squares

To predict value of Y if you know X.

Residual: The vertical distance from the horizontal axis to any point, can be decomposed into two parts: the vertical distance from the horizontal axis to the line, and the vertical distance from the line to the point. The first of these is called the fitted value,
and the second is called the residual. So a fitted value is the predicted value of the dependent variable. The corresponding residual is the difference between the actual and fitted values of the dependent variable.

```python
import math

# set the minimum and maximum value of the x- and y-axis
xmin = math.floor(male_chinstrap['flipper_length_mm'].min() / 10) * 10
xmax = math.ceil(male_chinstrap['flipper_length_mm'].max() / 10) * 10

ymin = math.floor(male_chinstrap['body_mass_g'].min() / 10) * 10
ymax = math.ceil(male_chinstrap['body_mass_g'].max() / 10) * 10

least_squares = pd.DataFrame({
        'x': male_chinstrap.flipper_length_mm,
        'y': male_chinstrap.body_mass_g
    })
mx = least_squares.x.mean()
my = least_squares.y.mean()

least_squares['(x-x̄)'] = least_squares['x'] - mx
least_squares['(y-ȳ)'] = least_squares['y'] - my

least_squares['(x-x̄)(y-ȳ)'] = least_squares['(x-x̄)'] * least_squares['(y-ȳ)']
least_squares['(x-x̄)²'] = least_squares['(x-x̄)'] ** 2
least_squares
```

```python
# Numerator and denomitator of the formula for b_0:
num = sum(least_squares['(x-x̄)(y-ȳ)'])
denom = sum(least_squares['(x-x̄)²'])
beta1 = num/denom
beta0 = my - beta1 * mx

print(f"beta_1 = {num:.4f} / {denom:.4f} = {beta1:.4f}")
print(f"beta_0 = {my:.4f} - {beta1:.4f} * {mx:.4f} = {beta0:.4f}")
print(f"ŷ = {beta0:.4f} + {beta1:.4f} x")
```

Bijhorende plot:

```python
x_values = [xmin, xmax]
y_values = [beta1 * x_values[0] + beta0, beta1 * x_values[1] + beta0]

sns.lineplot(x=x_values, y=y_values);
sns.scatterplot(x=male_chinstrap.flipper_length_mm, y=male_chinstrap.body_mass_g);
```

```python
# Met korte formule
from sklearn.linear_model import LinearRegression

male_chinstrap_x = male_chinstrap.flipper_length_mm.values.reshape(-1,1)
male_chinstrap_y = male_chinstrap.body_mass_g

weight_model = LinearRegression().fit(male_chinstrap_x, male_chinstrap_y)

print(f"Regression line: ŷ = {weight_model.intercept_:.4f} + {weight_model.coef_[0]:.4f} x")
```

```python
#Methode voor B0 en B1
x = cats[cats['Sex'] == 'F']['Hwt']
y = cats[cats['Sex'] == 'F']['Bwt']
a, b = np.polyfit(x, y, 1)
print(f' y = {a} * x + {b}')
```

#### Covariance

Covariance is a measure that indicates whether a (linear) relationship
between two variables is increasing or decreasing.

Note: Covariance of population (denominator 𝑛) vs. sample (denominator 𝑛 − 1)

Cov > 0: increasing  
Cov ≈ 0: no relationship  
Cov < 0: decreasing

Covariance kan serieus verschillen adhv welke meeteeinheid je gebruikt. Cov van 2500 in g, is maar 2.5 in kg.

```python
# Calculate covar using the formula
covar = sum((male_chinstrap.flipper_length_mm - male_chinstrap.flipper_length_mm.mean()) * 
            (male_chinstrap.body_mass_g - male_chinstrap.body_mass_g.mean())) / (len(male_chinstrap) - 1)
        
print(f"Cov(x,y) = {covar}")
```

```python
# Met functie
np.cov(
    male_chinstrap.flipper_length_mm,
    male_chinstrap.body_mass_g,
    ddof=1)[0][1]
```

#### Pearson's product-moment corelation coefficient

𝑅 is a measure for
the strength of a linear correlation between 𝑥 and 𝑦.

Correlation is a unitless quantity that is unaffected by the measurement scale. For example, the correlation
is the same regardless of whether the variable X (e.g. height of a person) is measured in millimeters, centimeters, decimeters or meters.

Always between -1 and 1. The closer to -1 or 1, the closer the points in a scatterplot are to a striaght line.

Bvb: R = (-)0.8 -> (decreasing)increasing and strong linear relation

```python
# Correlation calculated from covariance
stdx = male_chinstrap.flipper_length_mm.std()
stdy = male_chinstrap.body_mass_g.std()

R1 = covar / (stdx * stdy)
print(f"R ≈ {covar:.4f} / ( {stdx:.4f} * {stdy:.4f} ) ≈ {R1:.4f}")
```

```python
# Correlation from elaborated formula
xx = male_chinstrap.flipper_length_mm - male_chinstrap.flipper_length_mm.mean()
yy = male_chinstrap.body_mass_g - male_chinstrap.body_mass_g.mean()
R2 = sum(xx * yy) / (np.sqrt(sum(xx ** 2) * sum(yy ** 2)))
print(f"R ≈ {R2:.4f}")
```

```python
# Python function numpy.corrcoef() - returns a matrix, like numpy.cov()
cor = np.corrcoef(
    male_chinstrap.flipper_length_mm,
    male_chinstrap.body_mass_g)[0][1]
print(f"R ≈ {cor:.4f}")
```

#### Coefficient of determenation

R**2: Zelfde als hierboven, maar ^2 -> makkelijker af te lezen.

| $abs(R)$  |  $R^2$   | Explained variance |   Linear relation    |
| :-------: | :------: | :----------------: | :------------------: |
|   < .3    |   < .1   |       < 10%        |      very weak       |
|  .3 - .5  | .1 - .25 |     10% - 25%      |         weak         |
|  .5 - .7  | .25 - .5 |     25% - 50%      |       moderate       |
| .7 - .85  | .5 - .75 |     50% - 75%      |        strong        |
| .85 - .95 | .75 - .9 |     75% - 90%      |     very strong      |
|   > .95   |   > .9   |       > 90%        | exceptionally strong |

Remark that the value of R doesn't say anything about the steepness of the regression line! It only indicates how close the observations are to the regression line. Therefore, it is wrong to say that a value of e.g. R = 0.8 indicates a strongly increasing linear relation! Instead, you should say it indicates an _increasing and strong linear relation_.

```python
# Bovenstaande formules voor R te berekenen, en dan
cor ** 2
```

```python
# OF
chinstrap_x = male_chinstrap.flipper_length_mm.values.reshape(-1,1)
chinstrap_y = male_chinstrap.body_mass_g

families_model = LinearRegression().fit(chinstrap_x, chinstrap_y)
families_model.score(chinstrap_x, chinstrap_y)
```

## 7. Time series analysis

## Extra

Drop all NAN data

```python
data = data.dropna()
```

Maak van datum in string formaat een echte datum

```python
data.datumKolom = pd.to_datetime(data.datumKolom, format='%Y/%m/%d')
```

Voornamelijk voor H7: Maak van datums de index

```python
data.set_index('datumKolom', inplace = True)
```
