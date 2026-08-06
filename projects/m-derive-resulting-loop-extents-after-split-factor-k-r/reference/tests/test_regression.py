from tvmutils.validator import trigger_vectorization_error, ScheduleError
import pytest


def test_vectorization_dependency_error():
    try:
        trigger_vectorization_error(16)
        assert False, "Expected ScheduleError"
    except ScheduleError:
        assert True
