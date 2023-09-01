import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib import rcParams

rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"] + rcParams["font.serif"]
rcParams["font.size"] = 17
rcParams["figure.figsize"] = (4, 3)
def plot_scala(Infile,OutFile):
    # Load data from Excel file
    df = pd.read_excel(Infile)

    # Extract the relevant data
    queries = df.columns[1:].tolist()  # Exclude the first column
    queries = ['Q0','Q1','Q2','Q3','Q4','Q5','Q6','Q7']
    with_restriction = df.iloc[0, 1:].tolist()  # Exclude the first element of the first row
    without_restriction = df.iloc[1, 1:].tolist()  # Exclude the first element of the second row

    # Plotting
    x = range(len(queries))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, with_restriction, width, label='With Restriction')
    rects2 = ax.bar([i + width for i in x], without_restriction, width, label='Without Restriction')

    # Add labels, title, and axes ticks
    ax.set_xlabel('Query')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(queries)
    ax.legend()
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))

    plt.savefig(OutFile, bbox_inches="tight")

plot_scala('cit-patents.xlsx','restriction_cit-Patents.pdf')
plot_scala('roadNet-PA.xlsx','restriction_roadNet-PA.pdf')