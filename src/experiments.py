import json
import os

from ucttp import _main

INPUT_DIR = "./../inputs"
PARAMETERS = "/parameters.json"
POPULATION_DEPENDENCE = "./../outputs/population_dependence.txt"
GENERATION_DEPENDENCE = "./../outputs/generation_dependence.txt"


def set_parameters(population_size, generations_number):
    parameters_file = os.path.abspath(os.path.dirname(__file__) + f'{PARAMETERS}')
    with open(parameters_file, 'r') as f:
        data = json.load(f)
        data['population_size'] = population_size
        data['generations_number'] = generations_number
    os.remove(parameters_file)
    with open(parameters_file, 'w') as f:
        json.dump(data, f, indent=4)


def population_size_dependence():
    result_file = os.path.abspath(os.path.dirname(__file__) + f'{POPULATION_DEPENDENCE}')
    _, _, instances = next(os.walk(os.path.abspath(os.path.dirname(__file__) + f'{INPUT_DIR}')))
    for instance in instances:
        for population_size in range(10, 101, 10):
            set_parameters(population_size, 100)
            print(f"{instance} - {population_size}")
            result = _main(os.path.abspath(os.path.dirname(__file__) + f'{INPUT_DIR}/{instance}'))
            with open(result_file, 'a') as file:
                file.write(f"{instance} - {population_size} - {result}\n")


def generations_number_dependence():
    result_file = os.path.abspath(os.path.dirname(__file__) + f'{GENERATION_DEPENDENCE}')
    _, _, instances = next(os.walk(os.path.abspath(os.path.dirname(__file__) + f'{INPUT_DIR}')))
    instances = ["large.tim"]
    for instance in instances:
        set_parameters(50, 20000)
        print(f"{instance}")
        result = _main(os.path.abspath(os.path.dirname(__file__) + f'{INPUT_DIR}/{instance}'))
        with open(result_file, 'a') as file:
            file.write(f"{instance} - {result}\n")


if __name__ == '__main__':
    population_size_dependence()
    generations_number_dependence()
