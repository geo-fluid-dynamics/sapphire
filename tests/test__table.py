import fempy.table


def test__table():

    table = fempy.table.Table(("foo", "bar"))

    print(str(table))
    