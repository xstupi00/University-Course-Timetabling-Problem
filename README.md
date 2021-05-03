# Simulation Tools and Techniques - Project
## University Course Timetabling Problem

##### Author: Stupinsky Simon <xstupi00@stud.fit.vutbr.cz>

This project is based on the following article:
> K. Shaker, S. Abdullah and A. Hatem, "A Differential Evolution Algorithm for the University course timetabling problem," 2012 4th Conference on Data Mining and Optimization (DMO), 2012, pp. 99-102, doi: 10.1109/DMO.2012.6329805.
Retrieved from: https://ieeexplore.ieee.org/document/6329805


## Usage
```python3 ucttp.py [OPTIONS]```

Options:
- -i, --instance FILE  The problem instance, as file name in the [input](./inputs) directory, with the specified format. [required]
- --help               Show this message and exit.

A [parameter](src/parameters.json) file includes parameter setting of Differential Evaluation Algorithm:

- generations_number: the number of generations (iterations of DEA) [default=20000] 
- population_size: the size of the initial population [default=50]
- crossover_rate: cross-over rate within DEA iterations [default=0.8] 
- mutation_rate: mutation rate within DEA iterations [default=0.5]
- n1_applications: the maximal number of applications n1 move [default=100] 
- n2_applications: the maximal number of applications n2 move [default=100]

The application provides a log statement of the progress of performed actions. 
To list them, it is necessary to set the logger level in the [file](src/ucttp.py) at line `9`.
We recommend level `logging.WARNING` to see the main progress when the algorithm is performed.

### Input
The program expects a problem instance given as input in format that can be found on site: 
http://sferics.idsia.ch/Files/ttcomp2002/IC_Problem/node7.html

### Output
The best solution found is outputted on the STDOUT in format described on site: 
http://sferics.idsia.ch/Files/ttcomp2002/IC_Problem/Output_format.htm

### Testing
Validity of results given by this application has been tested using an official 
solution checking tool available on site: 
http://sferics.idsia.ch/Files/ttcomp2002/IC_Problem/Checking_solutions.htm

## Experiments

To make the program with concrete instance you can use both following commands:

```shell
make run INSTANCE=small1.tim
python3 src/ucttp.py --instance small1.tim
```

To collect data from described experiments you can run the following command.
Please note that this collection takes some time, and the collected data is therefore available in the [output](outputs) directory.

```shell
make experiments
```

To plot graphs from the collected data you can run the following command.
It creates the PDF files in the [output](outputs) directory with the relevant presented graphs.

```shell
make graph
```

## Project Structure
- `doc/doc.pdf` - project documentation
- `inputs/` - input datasets
- `outputs/` - results of experiments
- `src/` - source files with a Python implementation
- `Makefile` - rules to run program and experiments
- `requirements.txt` - specifying python packages that are required to run the project