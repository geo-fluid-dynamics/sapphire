import typing
import firedrake as fe


def verify_function_values_at_points(
        function: fe.Function,
        points: typing.Union[typing.Tuple[float], typing.Tuple[typing.Tuple[float]]],
        expected_values: typing.Union[typing.Tuple[float], typing.Tuple[typing.Tuple[float]]],
        absolute_tolerances: typing.Union[typing.Tuple[float], typing.Tuple[typing.Tuple[float]]]):

    if len(expected_values) != len(points) or (len(expected_values) != len(absolute_tolerances)):

        raise Exception("There must be an expected value and a tolerance for each point.")

    for point, expected_value, tolerance in zip(points, expected_values, absolute_tolerances):

        value = function.at(point)

        if isinstance(value, float):

            value = (value,)

            expected_value = (expected_value,)

        print("Expected {} and found {}.".format(expected_value, value))

        for i, v_i in enumerate(value):

            if expected_value[i] is None:

                continue

            error = abs(v_i - expected_value[i])

            if error > tolerance[i]:

                raise Exception("Absolute error ({}) is greater than tolerance ({})".format(error, tolerance[i]))
