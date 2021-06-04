import sapphire.examples.heat_driven_cavity_with_water


def test__heat_driven_cavity_with_water(tmpdir):

    sapphire.examples.heat_driven_cavity_with_water.verify_default_simulation()
