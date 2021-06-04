import sapphire.examples.heat_driven_cavity


def test__heat_driven_cavity(tmpdir):

    sapphire.examples.heat_driven_cavity.run_steady_state_heat_driven_cavity_simulation()
