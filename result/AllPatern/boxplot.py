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
            value1 = float(df1.iloc[j, 2*i-1])
            value2 = float(df2.iloc[j, 2*i-1].split('ms')[0])
            ratio.append(value2 / value1)
    ratios.append(ratio)
    print(ratio)
# ratio = df1[column2] / df2[column1]

# df = pd.DataFrame(ratios)
# df.plot.box(title="")
# plt.grid(linestyle="--", alpha=0.3)
# plt.show()
# plt.savefig('box_plot.png')

# for i in ratios[6]:
    # print(i)
print(pd.DataFrame(ratios[6]).describe())
print(pd.DataFrame(ratios[7]).describe())
# print(len(ratios[6]))
# print(len(ratios[7]))
# plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))

mpl.rcParams['font.size'] = 12
plt.boxplot(ratios,showfliers=False)
plt.xlabel('Pattern')
plt.ylabel('Speed up')
plt.ylim(0, 40)
plt.xticks(range(1, 9))
plt.show()
plt.savefig('box_plot.pdf')