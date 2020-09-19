import sapphire.output


def test__table():

    table = sapphire.output.Table(("foo", "bar"))

    print(str(table))
    