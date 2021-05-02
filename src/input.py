import logging
import os

from ucttp_instance import UcttpInstance

logger = logging.getLogger()

INPUT_DIR = '/../inputs'  # input directory


def get_input_file(_, __, file_name: str) -> str:
    """
    Retrieves an input file name with a given problem.
    :param click.Context _: called context of the process
    :param click.Option __: parameter that is being parsed and read from commandline
    :param str file_name: A name of an instance of the problem.
    :return str: An input file name with a given problem.
    :raises: FileNotFoundError if an input file has not been found.
    """
    global INPUT_DIR

    file_path = os.path.abspath(os.path.dirname(__file__) + f'{INPUT_DIR}/{file_name}')
    if not os.path.exists(file_path):
        raise FileNotFoundError

    return file_path


def valid_input_data(events, rooms, features, students, room_sizes, rooms_features, students_events, events_features):
    assert events > 0 and rooms > 0 and features > 0 and students > 0
    assert all(all(room_feature in [True, False] for room_feature in room_features) for room_features in rooms_features)
    assert all(len(room_features) == features for room_features in rooms_features)
    assert len(rooms_features) == rooms
    assert all(
        all(student_event in [True, False] for student_event in student_events) for student_events in students_events
    )
    assert all(len(student_events) == events for student_events in students_events)
    assert len(students_events) == students
    assert all(
        all(event_feature in [True, False] for event_feature in event_features) for event_features in events_features
    )
    assert all(len(event_features) == features for event_features in events_features)
    assert len(events_features) == events
    assert all(room_size > 0 for room_size in room_sizes)
    assert len(room_sizes) == rooms

    logger.debug(f"Events: {events}, Rooms: {rooms}, Features: {features}, Students: {students}")
    logger.debug(f"Room Sizes: {room_sizes}")
    logger.debug(f"Events per Student: {students_events}")
    logger.debug(f"Features per Room: {rooms_features}")
    logger.debug(f"Features per Event: {events_features}")


def parse_input_file(file_path: str):
    """
    Parses an input file.
    :param str file_path: A name of an input file.
    :return:
    """
    room_sizes = []
    students_events, student_events = [], []
    rooms_features, room_features = [], []
    events_features, event_features = [], []

    with open(file_path) as lines:
        for idx, line in enumerate(lines):
            if idx == 0:
                events, rooms, features, students = [int(x) for x in line.split()]
            elif idx <= rooms:
                room_sizes.append(int(line))
            elif idx <= rooms + (events * students):
                assert int(line) in [0, 1]
                student_events.append(bool(int(line)))
                if (idx - rooms) % events == 0:
                    students_events.append(student_events)
                    student_events = []
            elif idx <= (rooms + (events * students) + (rooms * features)):
                assert int(line) in [0, 1]
                room_features.append(bool(int(line)))
                if (idx - (rooms + (events * students))) % features == 0:
                    rooms_features.append(room_features)
                    room_features = []
            elif idx <= (rooms + (events * students) + (rooms * features) + (events * features)):
                assert int(line) in [0, 1]
                event_features.append((bool(int(line))))
                if (idx - (rooms + (events * students) - (rooms * features))) % features == 0:
                    events_features.append(event_features)
                    event_features = []

        valid_input_data(
            events, rooms, features, students, room_sizes, rooms_features, students_events, events_features
        )

        return UcttpInstance(
            events, rooms, features, students, room_sizes, students_events, rooms_features, events_features
        )
