import itertools
import json
import logging
import os

from functools import reduce
from types import SimpleNamespace

import jsonschema
import numpy as np
import numpy.random

from ucttp_instance import UcttpInstance

import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

logger = logging.getLogger()

PARAMETERS_FILE = "parameters.json"

PARAMETERS_SCHEMA = {
    "generations_number": {"type": "integer", "minimum": 0},
    "population_size": {"type": "integer", "minimum": 0},
    "crossover_rate": {"type": "number", "minimum": 0, "maximum": 1},
    "mutation_rate": {"type": "number", "minimum": 0, "maximum": 1},
    "n1_applications": {"type": "integer", "minimum": 0},
    "n2_applications": {"type": "integer", "minimum": 0},
}


class DifferentialEvolution:
    def __init__(self, instance: UcttpInstance):
        self._instance = instance
        self._parameters = None
        self._collision_events = None
        self._sorted_events = None
        self._fitness = None
        self._students_events = None
        self._event_timetable = self._init_event_timetable()
        self._event_timetables = []
        self._DAYS = 5
        self._HOURS = 9
        self.TIMESLOTS = self._DAYS * self._HOURS

    def _init_event_timetable(self):
        return np.array([(-1, -1)] * self._instance.events, dtype=np.dtype('int,int'))

    def _load_parameters(self):
        global PARAMETERS_FILE

        file = os.path.abspath(os.path.dirname(__file__) + f"/{PARAMETERS_FILE}")
        if not os.path.exists(file):
            raise FileNotFoundError
        with open(file) as json_file:
            parameters = json.load(json_file)
            jsonschema.validate(parameters, schema=PARAMETERS_SCHEMA)
            self._parameters = json.loads(json.dumps(parameters), object_hook=lambda d: SimpleNamespace(**d))

        logger.debug(f"Generations number: {self._parameters.generations_number}")
        logger.debug(f"Population size: {self._parameters.population_size}")
        logger.debug(f"Crossover rate: {self._parameters.crossover_rate}")
        logger.debug(f"Mutation rate: {self._parameters.mutation_rate}")

    def _constructive_heuristic(self):
        self._students_events = self._instance.get_student_events().to_dict()
        self._sorted_events, self.available_rooms = self._instance.get_sorted_events()
        self._collision_events = self._instance.get_collision_events()
        for idx in range(self._parameters.population_size):
            feasible = self._construct_event_timetable()
            logger.warning(f"Constructed {'feasible' if feasible else 'UNFEASIBLE'} solution no. {idx}.")
        self._event_timetables = np.stack(self._event_timetables, axis=0)

    def set_event_timetable(self, timetables, event):
        event_timetable = np.random.choice(timetables, 1)[0]
        logger.info(f"Event {event} planned for: {event_timetable}")
        self._event_timetable[event] = event_timetable

    def _construct_event_timetable(self):
        self._event_timetable = self._init_event_timetable()
        for idx, event in enumerate(self._sorted_events):
            logger.info(f"Available rooms for event {event}: {self.available_rooms[event]}")
            timetables = self._get_timetables(event, idx)
            if timetables.size > 0:
                self.set_event_timetable(timetables, event)
            else:
                missing_events = np.where(self._event_timetable['f1'] == -1)
                logger.warning(f"No timeslot for event {event}; missing {missing_events[0].size}")
                success = self._apply_neighbourhood_moves(event, idx)
                if not success:
                    return False
        self._event_timetables.append(self._event_timetable)
        return True

    def _apply_neighbourhood_moves(self, event, iteration):
        def perform_move_iterations(applications, neighbourhood_move):
            for idx, _ in enumerate(range(applications)):
                neighbourhood_move()
                timetables = self._get_timetables(event, iteration)
                if timetables.size > 0:
                    logger.info(f"{neighbourhood_move.__name__[-2:]} was successful at {idx} iter: {timetables.size}.")
                    self.set_event_timetable(timetables, event)
                    return True
            return False

        n1_success = perform_move_iterations(self._parameters.n1_applications, self._apply_move_n1)
        # if self._event_timetable[event]['f0'] == -1 and self._event_timetable[event]['f1'] == -1:
        if not n1_success:
            n2_success = perform_move_iterations(self._parameters.n2_applications, self._apply_move_n2)
        return n1_success if n1_success else n2_success

    def _apply_move_n2(self, event_timetable=None):
        def get_collisions(e1, e2):
            return np.intersect1d(np.take(event_timetable, np.concatenate([
                self._collision_events[self._collision_events['f0'] == e1]['f1'],
                self._collision_events[self._collision_events['f1'] == e1]['f0']
            ]))['f1'], event_timetable[event_timetable['f1'] == event_timetable[e2]['f1']]['f1'])

        event_timetable = self._event_timetable if event_timetable is None else event_timetable
        events = np.random.choice(np.where(event_timetable['f0'] == np.random.choice(
            np.unique(event_timetable[event_timetable['f0'] != -1]['f0'])
        ))[0], size=2)
        if get_collisions(events[0], events[1]).size == 0 and get_collisions(events[1], events[0]).size == 0:
            event_timetable[events[0]], event_timetable[events[1]] = \
                event_timetable[events[1]], event_timetable[events[0]]
        return event_timetable

    def _apply_move_n1(self, event_timetable=None):
        event_timetable = self._event_timetable if event_timetable is None else event_timetable
        event = np.random.choice(np.where(event_timetable['f0'] != -1)[0], 1)[0]
        event_timetable[event] = (-1, -1)
        timetables = self._get_timetables(event, -1, final=False)
        if timetables.size > 0:
            timetables = timetables.view(int).reshape(timetables.shape + (-1,))
            timetable = timetables[np.argmin(
                np.apply_along_axis(lambda x: self._object_function(event_timetable, x, event), 1, timetables)
            )]
            event_timetable[event] = (timetable[0], timetable[1])
        return event_timetable

    @staticmethod
    def _construct_sequences(day_timeslots):
        sequences, sequence = [], []
        timeslots_len = len(day_timeslots)
        for idx in range(timeslots_len):
            sequence.append(day_timeslots[idx])
            if idx >= timeslots_len - 1 or day_timeslots[idx] + 1 != day_timeslots[idx + 1]:
                sequences.append(sequence)
                sequence = []
        return sequences

    def _object_function(self, event_timetable, timetable=None, event=None):
        def get_consecutive_score(events):
            sched_timeslots = event_timetable[events]['f1'].tolist()
            days_timeslots = []
            [days_timeslots.append([]) for _ in range(5)]
            [days_timeslots[d].append(s % self._HOURS)
             for d in range(0, self._DAYS) for s in sched_timeslots if s // self._HOURS == d]

            return sum([sum([max(len(seq) - 2, 0)
                             for seq in self._construct_sequences(day_timeslot)]) for day_timeslot in days_timeslots]), \
                   [[t // self._HOURS for t in sched_timeslots].count(d) for d in range(self._HOURS)].count(1)

        if event is not None and timetable is not None:
            event_timetable[event] = (timetable[0], timetable[1])

        s1 = self._instance.get_events_sizes()[np.where(
            (event_timetable['f1'] % self._HOURS == self._HOURS - 1) & (event_timetable['f1'] != -1)
        )].sum()

        s2_s3 = [get_consecutive_score(student_events[1])
                 for student_events in self._students_events['Event'].items() if len(student_events[1]) > 2]

        s2 = sum(s2 for s2, _ in s2_s3)
        s3 = sum(s3 for _, s3 in s2_s3)
        logger.info(f"S1={s1}; S2={s2}, S3={s3}")
        return s1 + s2 + s3

    def _get_timetables(self, event, iteration, final=True):
        timetables = np.array([
            timeslot for timeslot in list(itertools.product(self.available_rooms[event], range(self.TIMESLOTS)))
        ], dtype=np.dtype('int, int'))
        # timetables = np.delete(timetables, np.where(timetables == np.intersect1d(timetables, self._event_timetable)))
        timetables = timetables[~np.in1d(timetables, self._event_timetable)]

        event_collisions = np.concatenate([
            self._collision_events[self._collision_events['f0'] == event]['f1'],
            self._collision_events[self._collision_events['f1'] == event]['f0']
        ])

        timetables = timetables[~np.in1d(timetables['f1'], np.take(self._event_timetable, event_collisions)['f1'])]
        assert np.intersect1d(timetables['f1'], np.take(self._event_timetable, event_collisions)['f1']).size == 0
        assert np.intersect1d(timetables, self._event_timetable).size == 0
        return self._get_final_timetable(timetables, iteration) if final and timetables.size > 0 else timetables

    def _get_final_timetable(self, timetables, iteration):
        def reducer(accumulator, element):
            for key, value in element.items():
                accumulator[key] = accumulator.get(key, 0) + value
            return accumulator

        slots_counts = dict(zip(*np.unique(
            self._event_timetable[np.in1d(self._event_timetable['f1'], timetables['f1'])]['f1'], return_counts=True
        )))
        slots_counts = reduce(reducer, [{slot: 0 for slot in timetables['f1']}, slots_counts], {})
        timetables = timetables[np.in1d(
            timetables['f1'],
            [i[0] for i in slots_counts.items() if i[1] == max(slots_counts.items(), key=lambda x: x[1])[1]])
        ]

        rooms_counts = np.fromiter(dict(zip(*np.unique(
            np.concatenate(
                np.take(self.available_rooms, np.take(self._sorted_events, range(iteration, self._instance.events)))
            ),
            return_counts=True
        ))).values(), dtype=int)
        min_requested_rooms = np.where(rooms_counts == np.amin(np.take(rooms_counts, np.unique(timetables['f0']))))[0]

        return timetables[
            np.in1d(timetables.view(dtype='int, int').reshape(timetables.shape[-1])['f0'], min_requested_rooms)
        ]

    def _evaluate_timetables(self):
        self._fitness = np.vectorize(self._object_function, signature='(n)->()')(self._event_timetables)
        self._event_timetables = self._event_timetables[np.argsort(self._fitness)]
        self._fitness = np.sort(self._fitness)

    def _mutation(self):
        p1, p2 = self._event_timetables[np.random.randint(0, self._event_timetables.shape[0], 2)]
        mutation_cond = numpy.random.uniform() <= self._parameters.mutation_rate
        return [self._apply_move_n1(p) if mutation_cond else self._apply_move_n2(p) for p in [p1, p2]]

    def _crossover(self, p1, p2):
        def process_parent(parent1, parent2):
            slot = np.random.choice(np.unique(parent1['f1']))
            free_rooms = np.in1d(p1[np.in1d(parent1['f1'], slot)], parent2)
            events = np.where(np.in1d(parent1['f1'], slot))[0]
            scheduled_collision = np.vectorize(lambda x: np.in1d(slot, np.unique(parent2[np.concatenate([
                self._collision_events[np.in1d(self._collision_events['f0'], x)]['f1'],
                self._collision_events[np.in1d(self._collision_events['f1'], x)]['f0']
            ])]['f1'])))(events)
            for idx, can_crossover in enumerate(np.stack([free_rooms, scheduled_collision]).all(axis=0)):
                if can_crossover and np.random.uniform() <= self._parameters.crossover_rate:
                    logger.info(f"Perform crossover on event {events[idx]}")
                    parent2[events[idx]] = parent1[events[idx]]
            return parent1, parent2

        p1, p2 = process_parent(p1, p2)
        p2, p1 = process_parent(p2, p1)

        return p1, p2

    def _differential_evolution_algorithm(self):
        p1_m, p2_m = self._mutation()
        child_1, child_2 = self._crossover(p1_m, p2_m)
        child_qualities = [self._object_function(child) for child in [child_1, child_2]]
        better_child = (child_qualities[0], 0) if child_qualities[0] < child_qualities[1] else (child_qualities[1], 1)
        if better_child[0] < self._fitness[0]:
            np.roll(self._event_timetables, 1)
            np.roll(self._fitness, 1)
            self._event_timetables[0] = child_2 if better_child[1] else child_1
            self._fitness[0] = better_child[0]

    def run(self):
        self._load_parameters()
        self._constructive_heuristic()
        self._evaluate_timetables()
        results = []  #
        for idx, _ in enumerate(range(self._parameters.generations_number)):
            logger.warning(f"Run DEA iteration no. {idx}: {self._fitness[0]}")
            results.append(self._fitness[0])  #
            if self._fitness[0] == 0:
                break
            self._differential_evolution_algorithm()
        return results  #
        # [print(str(x[0]) + ' ' + str(x[1])) for x in self._event_timetables[0]] #
