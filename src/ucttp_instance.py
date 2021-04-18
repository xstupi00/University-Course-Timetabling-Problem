import itertools
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger()


class UcttpInstance:
    def __init__(self, events, rooms, features, students, room_sizes, students_events, rooms_features, events_features):
        self._events = events
        self._rooms = rooms
        self._features = features
        self._students = students
        self._room_sizes = np.array(room_sizes)
        self._students_events = np.array(students_events)
        self._rooms_features = np.array(rooms_features)
        self._events_features = np.array(events_features)

    @property
    def events(self):
        return self._events

    def get_events_sizes(self):
        return self._students_events.T.sum(axis=1)

    def get_student_events(self):
        return pd.DataFrame(
            np.argwhere(self._students_events), columns=['Student', 'Event']
        ).groupby('Student').aggregate({'Event': lambda x: list(x)})

    def get_sorted_events(self):
        sorted_events = [self.get_rooms_for_event(event) for event in range(self._events)]
        return np.argsort([event.size for event in sorted_events]), sorted_events

    def get_rooms_for_event(self, event: int):
        return np.intersect1d(
            np.where(self._room_sizes >= self._students_events.sum(axis=0)[event]),
            np.where(np.all(np.logical_or(np.logical_not(self._events_features[event]), self._rooms_features), axis=1))
        )

    def get_collision_events(self):
        collision_events = set()
        df = self.get_student_events()
        df = df[df['Event'].map(len) >= 2].apply(lambda x: [x for x in itertools.combinations(x['Event'], 2)], axis=1)
        [[collision_events.add(x) for x in val] for val in df.values]
        logger.debug(f"Collision Events: {collision_events}")
        return np.array(list(collision_events), dtype=np.dtype('int,int'))
