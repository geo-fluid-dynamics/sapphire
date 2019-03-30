import sunfire.table


def test__table():

    table = sunfire.table.Table(("foo", "bar"))

    print(str(table))
    