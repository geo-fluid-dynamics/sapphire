import fem.table


def test__table():

    table = fem.table.Table(("foo", "bar"))

    print(str(table))
    