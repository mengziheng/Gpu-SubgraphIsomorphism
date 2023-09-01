import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib import rcParams

path = "/home/zhmeng/GPU/Gpu-SubgraphIsomorphism/result/Scalability/"
rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"] + rcParams["font.serif"]
rcParams["font.size"] = 20
rcParams["figure.figsize"] = (4, 3)


def plot_scala(Infile, OutFile):
    # Read data from Excel file
    data = pd.read_excel(Infile, index_col=0)
    data = data.iloc[0] / data
    print(data)
    # Extract the query names and GPU counts from the DataFrame
    query_names = list(data.columns)
    gpu_counts = data.index.tolist()

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot each query
    for query in query_names:
        plt.plot(gpu_counts, data[query], marker="o", label=query)

    # Set labels and title
    plt.xlabel("Number of GPUs")
    plt.ylabel("Speedup")
    plt.xscale("log")
    plt.yscale("log", base=2)
    # Set x-axis ticks
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_major_locator(LogLocator(base=2.0, numticks=12))

    # Set y-axis tick labels
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().yaxis.set_major_locator(LogLocator(base=2.0, numticks=12))

    plt.xticks(gpu_counts)

    # Set legend
    plt.legend()

    # Show the plot
    plt.grid(False)
    plt.savefig(OutFile, bbox_inches="tight")


# plot_scala(path + "cit-Patents.xlsx", path + "scalability_cit-Patents.pdf")
# plot_scala(path + "roadNet-PA.xlsx", path + "scalability_roadNet-PA.pdf")
# plot_scala(path + "P1a.xlsx", path + "P1a.pdf")
plot_scala(path + "201512020000.xlsx", path + "201512020000.pdf")
