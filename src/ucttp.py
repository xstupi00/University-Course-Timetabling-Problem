import logging
import sys

import click

from differential_evolution import DifferentialEvolution
from input import get_input_file, parse_input_file

LOGGING_LEVEL = logging.CRITICAL


def setup_logger():
    """
    Setup routine for logging.
    :return:
    """
    root = logging.getLogger()
    root.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(LOGGING_LEVEL)
    ch.setFormatter(formatter)
    root.addHandler(ch)


@click.command()
@click.option('--instance', '-i', required=True, type=click.STRING, callback=get_input_file,
              help="The problem instance files with the specified format.")
def main(instance):
    setup_logger()
    DifferentialEvolution(parse_input_file(instance)).run()


def _main(instance):
    setup_logger()
    return DifferentialEvolution(parse_input_file(instance)).run()


if __name__ == '__main__':
    main()
