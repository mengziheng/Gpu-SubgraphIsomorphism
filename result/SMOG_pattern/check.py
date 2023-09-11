import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl


excel_file = 'AllPatern.xlsx'
df1 = pd.read_excel(excel_file, sheet_name='SMOG')
df2 = pd.read_excel(excel_file, sheet_name='rps')

print(df1.shape[0],df1.shape[1])
column1 = 'Q1 time'
column2 = 'pattern-1-time'
ratios = []
for i in range(1, 9):
    ratio = []
    for j in range(len(df1)):
        if(df1.iloc[j, 2*i-1] != "#OT" and df2.iloc[j, 2*i-1] != "#OT"):
            value1 = float(df1.iloc[j, 2*i])
            value2 = float(df2.iloc[j, 2*i])
            if(value1 != value2):
                print(value1,value2,i,j)
