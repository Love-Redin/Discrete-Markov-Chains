# Example of usage

This example assumes familiarity with pandas.

Say that we have a pandas dataframe of daily rain data, looking like this:

```
      Representativt dygn  Nederbördsmängd
0              1947-01-01              0.0
1              1947-01-02              0.5
2              1947-01-03              0.0
3              1947-01-04              0.0
4              1947-01-05              0.0
                  ...              ...
27785          2023-01-27              0.0
27786          2023-01-28              0.0
27787          2023-01-29              0.1
27788          2023-01-30              0.0
27789          2023-01-31              0.0

[27790 rows x 2 columns]
```

Suppose further that the column "Nederbördsmängd" is given in millimeters per day and looks something like this:

```
df["Nederbördsmängd"].describe()
Out[123]: 
count    27790.000000
mean         1.483170
std          3.526825
min          0.000000
25%          0.000000
50%          0.000000
75%          1.300000
max         59.800000
Name: Nederbördsmängd, dtype: float64
```

In this case, we have a time series that can be modeled as a Markov Chain, by creating a function to map rain amount to one of four states: None, Low, Medium or High.

```
# Define a function to apply the conditions
def classify_rain(x):
    if x == 0:
        return 'None'
    elif 0 < x <= 5:
        return 'Low'
    elif 5 < x <= 15:
        return 'Medium'
    else:
        return 'High'
        
df['Rain amount'] = df['Nederbördsmängd'].apply(classify_rain)
rain_amount = list(df['Rain amount'])
```

If we now call our function transition_matrix with degree one, it looks like this:

```
transition_matrix(rain_amount, 1).reindex(["None", "Low", "Medium", "High"], axis=0).reindex(["None", "Low", "Medium", "High"], axis=1)
Out[128]: 
            None       Low    Medium      High
None    0.685661  0.254704  0.052544  0.007090
Low     0.396461  0.484926  0.102698  0.015914
Medium  0.288036  0.536343  0.144018  0.031603
High    0.260274  0.490411  0.175342  0.073973
```

If we do the same for degree two, it looks like this:

```
transition_matrix(rain_amount, 2).reindex(["None", "Low", "Medium", "High"], axis=1)
Out[131]: 
                   None       Low    Medium      High
High-High      0.370370  0.444444  0.111111  0.074074
High-Low       0.446927  0.396648  0.134078  0.022346
High-Medium    0.312500  0.578125  0.062500  0.046875
High-None      0.652632  0.273684  0.052632  0.021053
Low-High       0.304348  0.478261  0.155280  0.062112
Low-Low        0.376885  0.508153  0.101916  0.013045
Low-Medium     0.286814  0.559192  0.125120  0.028874
Low-None       0.603839  0.315881  0.071553  0.008726
Medium-High    0.171429  0.500000  0.242857  0.085714
Medium-Low     0.392256  0.462963  0.126263  0.018519
Medium-Medium  0.257053  0.542320  0.166144  0.034483
Medium-None    0.556426  0.324451  0.103448  0.015674
None-High      0.224299  0.514019  0.177570  0.084112
None-Low       0.420395  0.466181  0.094953  0.018470
None-Medium    0.300126  0.500631  0.166456  0.032787
None-None      0.725718  0.226442  0.042041  0.005799
```

From these transition matrices, we can estimate the probabilites of a given amount of rain tomorrow, given either the last day or the past two days.

