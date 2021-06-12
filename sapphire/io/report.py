from pathlib import Path
from csv import DictWriter
from datetime import datetime
from sapphire.data.solution import Solution


def report(
        solution: Solution,
        filepath_without_extension: str
        ):

    solution_dict = solution.__dict__.copy()

    rowdict = {'datetime': str(datetime.now())}

    for key in solution_dict.keys():

        value = solution_dict[key]

        if isinstance(value, float) or isinstance(value, int) or isinstance(value, str):

            rowdict[key] = value

    ufl_constants_dict = solution.ufl_constants._asdict()

    for key in ufl_constants_dict:

        rowdict[key] = ufl_constants_dict[key].__float__()

    filepath = filepath_without_extension + '.csv'

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    print("Appending row to report " + filepath)

    with open(filepath, 'a+') as file:

        writer = DictWriter(file, fieldnames=rowdict.keys())

        if Path(filepath).stat().st_size == 0:

            writer.writeheader()

        writer.writerow(rowdict)
