import firedrake as fe 
import fempy.mms
import fempy.benchmarks.melt_octadecane_in_cavity
import fempy.benchmarks.freeze_water_in_cavity


def test__validate__melt_octadecane():
    
    tf = 80.
    
    th = 40.
    
    h = -0.02
    
    
    s = 1./64.
    
    tau = 1.e-12
    
    
    rx = 2
    
    nx = 32
    
    
    rt = 2
    
    nt = 8
    
    
    q = 4
    
    
    expected_liquid_area = 0.61
    
    tolerance = 0.01
    
    
    model = fempy.benchmarks.melt_octadecane_in_cavity.Model(
        quadrature_degree = q,
        element_degree = rx - 1,
        time_stencil_size = rt + 1,
        meshsize = nx)
    
    model.timestep_size.assign(tf/float(nt))
    
    model.smoothing.assign(s)
    
    model.solid_velocity_relaxation_factor.assign(tau)
    
    model.topwall_heatflux_switchtime = th
    
    model.topwall_heatflux_postswitch = h
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "melt_octadecane/" 
        + "th{0}_h{1}/".format(th, h)
        + "s{0}_tau{1}/".format(s, tau)
        + "rx{0}_nx{1}_rt{2}_nt{3}/".format(rx, nx, rt, nt)
        + "q{0}/".format(q))
    
    model.smoothing_sequence = (0.5, s)
    
    model.solutions, _, = model.run(
        endtime = tf,
        topwall_heatflux_poststart = h,
        topwall_heatflux_starttime = th,
        plot = fempy.models.convection_coupled_phasechange.plot,
        report = True,
        write_solution = True)
    
    liquid_area = fempy.models.convection_coupled_phasechange.postprocess(
        model)["liquid_area"]
    
    print("Liquid area = {0}".format(liquid_area))
    
    assert(abs(liquid_area - expected_liquid_area) < tolerance)
    

def test__validate__freeze_water():
    
    mu_l__SI = 8.90e-4  # [Pa s]
    
    rho_l__SI = 999.84  # [kg / m^3]
    
    nu_l__SI = mu_l__SI/rho_l__SI  # [m^2 / s]
    
    t_f__SI = 2340.  # [s]
    
    L__SI = 0.038  # [m]
    
    Tau = pow(L__SI, 2)/nu_l__SI
    
    t_f = t_f__SI/Tau
    
    """ For Kowalewski's water freezing experiment,
    at t_f__SI 2340 s, t_f = 1.44.
    """
    dim = 2
    
    
    s = 1./200.
    
    tau = 1.e-12
    
    
    rx = 2
    
    nx = 32
    
    
    rt = 2
    
    nt = 4
    
    
    q = 4
    
    
    expected_liquid_area = 0.69
    
    tolerance = 0.01
    
    
    model = fempy.benchmarks.freeze_water_in_cavity.Model(
        quadrature_degree = q,
        element_degree = rx - 1,
        time_stencil_size = rt + 1,
        spatial_dimensions = dim,
        meshsize = nx)
    
    model.timestep_size = model.timestep_size.assign(t_f/float(nt))
    
    model.solid_velocity_relaxation_factor = \
        model.solid_velocity_relaxation_factor.assign(tau)
    
    model.smoothing = model.smoothing.assign(s)
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "freeze_water/"
        + "dim{0}".format(dim)
        + "s{0}_tau{1}/".format(s, tau)
        + "rx{0}_nx{1}_rt{2}_nt{3}/".format(rx, nx, rt, nt)
        + "q{0}/".format(q))
    
    model.solutions, _, = model.run(
        endtime = t_f,
        write_solution = True,
        plot = fempy.models.convection_coupled_phasechange.plot,
        report = True)
    
    liquid_area = fempy.models.convection_coupled_phasechange.postprocess(
        model)["liquid_area"]
    
    print("Liquid area = {0}".format(liquid_area))
    
    assert(abs(liquid_area - expected_liquid_area) < tolerance)
    