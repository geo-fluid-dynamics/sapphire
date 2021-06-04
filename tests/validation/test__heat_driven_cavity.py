import sapphire.examples.heat_driven_cavity


def test__heat_driven_cavity(tmpdir):

    sapphire.examples.heat_driven_cavity.verify_default_simulation()
