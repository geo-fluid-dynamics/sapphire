import csv
import datetime
import firedrake as fe
from sapphire.data import Simulation


def default_report(
        sim: Simulation,
        output_directory_path: str
        ):

    dictionary = sim.__dict__.copy()

    for key in dictionary.keys():

        if isinstance(dictionary[key], fe.Constant):

            dictionary[key] = dictionary[key].__float__()

    dictionary["datetime"] = str(datetime.datetime.now())

    with output_directory_path.joinpath("report").with_suffix(".csv").open("a+") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=dictionary.keys())

        if len(list(writer)) == 0:

            writer.writeheader()

        writer.writerow(dictionary)
