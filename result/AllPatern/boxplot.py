import pandas as pd
from matplotlib.ticker import ScalarFormatter
import sys
import os
import numpy as np
import time
from tkinter import font
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox
from matplotlib import markers, pyplot as plt, scale
import matplotlib
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
import math

rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"] + rcParams["font.serif"]
rcParams["font.size"] = 17
rcParams["figure.figsize"] = (4, 3)
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
        if(j == 14 or j == 15 or j == 17):
            continue
        if(df1.iloc[j, 2*i-1] != "#OT" and df2.iloc[j, 2*i-1] != "#OT"):
            value1 = float(df1.iloc[j, 2*i-1])
            if("ms" in str(df2.iloc[j, 2*i-1])):
                value2 = float(df2.iloc[j, 2*i-1].split('ms')[0])
            else:
                value2 = float(df2.iloc[j, 2*i-1])
            print(value1,value2)
            if(not math.isnan(value2) and not math.isnan(value1)):
                ratio.append(value2 / value1)
    if(i != 6):
        ratios.append(ratio)
    # print(len(ratio))
    # print(ratio)
# ratio = df1[column2] / df2[column1]

# df = pd.DataFrame(ratios)
# df.plot.box(title="")
# plt.grid(linestyle="--", alpha=0.3)
# plt.show()
# plt.savefig('box_plot.png')

# for i in ratios[6]:
    # print(i)
# print(pd.DataFrame(ratios[6]).describe())
# print(pd.DataFrame(ratios[7]).describe())
# print(len(ratios[6]))
# print(len(ratios[7]))
# plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))
fig, ax = plt.subplots()
ax.boxplot(ratios,showfliers=False)
plt.xlabel('Pattern')
plt.ylabel('Speedup')
# plt.yscale('log')

plt.ylim(0, 15)
ax.set_xticklabels(['Q0','Q1','Q2','Q3','Q4','Q5','Q6'])
plt.savefig('box_plot.png', bbox_inches="tight")
