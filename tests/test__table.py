import sapphire.table


def test__table():

    table = sapphire.table.Table(("foo", "bar"))

    print(str(table))
    