from pandas import DataFrame
df = DataFrame()
ls = [x for x in range(10)]
df['t'] = ls
df['t-1'] = df['t'].shift(1)
df['t+1'] = df['t'].shift(-1)
print(df)