import os
import matplotlib.pyplot as plt

from itertools import islice

POPULATION_DEPENDENCE = "./../outputs/population_dependence.txt"


def load_data():
    data_file = os.path.abspath(os.path.dirname(__file__) + f'{POPULATION_DEPENDENCE}')
    results = {}
    with open(data_file, 'r') as f:
        for _ in enumerate(range(0, 110, 10)):
            for idx, line in enumerate(islice(f, 10)):
                if idx == 0:
                    results[line.split('-')[0].strip()] = []
                results[line.split('-')[0].strip()].append(float(line.split('-')[2].strip()))
    return results


def construct_graph_small():
    data = load_data()
    for group in ['small', 'medium', 'large']:
        for k in range(1, 6):
            name = f"{group}{k}" if group != "large" else f"{group}"
            y = data.get(name + ".tim", [10] * 10)
            plt.plot(range(10, 101, 10), y, label=name, linewidth=2)
        plt.ylabel('Population Fitness', fontweight='bold', fontsize=14)
        plt.xlabel('Population Size', fontweight='bold', fontsize=14)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=3)
        plt.savefig(f"{group}.pdf")
        plt.cla()


if __name__ == '__main__':
    construct_graph_small()
