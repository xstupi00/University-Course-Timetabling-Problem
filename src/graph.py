import os
import matplotlib.pyplot as plt

from itertools import islice

POPULATION_DEPENDENCE = "./../outputs/population_dependence.txt"
GENERATION_DEPENDENCE = "./../outputs/generation_dependence.txt"



def load_population_data():
    data_file = os.path.abspath(os.path.dirname(__file__) + f'{POPULATION_DEPENDENCE}')
    results = {}
    with open(data_file, 'r') as f:
        for _ in enumerate(range(0, 110, 10)):
            for idx, line in enumerate(islice(f, 10)):
                if idx == 0:
                    results[line.split('-')[0].strip()] = []
                results[line.split('-')[0].strip()].append(float(line.split('-')[2].strip()))
    return results


def load_generation_data():
    data_file = os.path.abspath(os.path.dirname(__file__) + f'{GENERATION_DEPENDENCE}')
    results = {}
    with open(data_file, 'r') as f:
        for line in f:
            list_result = line.split('-')[1].strip()
            results[line.split('-')[0].strip()] = list(map(float, list_result[1:-1].split(',')))
    return results


def construct_graphs_population():
    data = load_population_data()
    for group in ['small', 'medium', 'large']:
        for k in range(1, 6):
            name = f"{group}{k}" if group != "large" else f"{group}"
            y = data.get(name + ".tim", [10] * 10)
            plt.plot(range(10, 101, 10), y, label=name, linewidth=2)
            if group == "large":
                break
        plt.ylabel('Population Fitness', fontweight='bold', fontsize=14)
        plt.xlabel('Population Size', fontweight='bold', fontsize=14)
        plt.legend(
            loc='lower center', bbox_to_anchor=(0.5, 0.05 if group != "medium" else 0.80),
            fancybox=True, shadow=True, ncol=3
        )
        plt.savefig(f"outputs/{group}_population.pdf")
        plt.cla()


def construct_graphs_generation():
    data = load_generation_data()
    for group in ['small', 'medium', 'large']:
        for k in range(1, 6):
            name = f"{group}{k}" if group != "large" else f"{group}"
            y = data.get(name + ".tim", [10] * 10)
            plt.plot(range(1, 20001, 1), y, label=name, linewidth=2)
            if group == "large":
                break
        plt.ylabel('Population Fitness', fontweight='bold', fontsize=14)
        plt.xlabel('Generation Number', fontweight='bold', fontsize=14)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.80), fancybox=True, shadow=True, ncol=3)
        plt.savefig(f"outputs/{group}_generation.pdf")
        plt.cla()


if __name__ == '__main__':
    construct_graphs_generation()
    construct_graphs_population()
