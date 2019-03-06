import firedrake as fe 
import fempy.mms
import fempy.benchmarks.melt_octadecane_in_cavity
import fempy.benchmarks.freeze_water_in_cavity


def test__regression__validate__melt_octadecane():
    
    endtime = 80.
    
    topwall_heatflux_switchtime = 40.
    
    q = -0.02
    
    
    s = 1./64.
    
    nx = 32
    
    Delta_t = 10.
    
    tau = 1.e-12
    
    
    expected_liquid_area = 0.61
    
    tolerance = 0.01
    
    
    model = fempy.benchmarks.melt_octadecane_in_cavity.Model(
        quadrature_degree = 4,
        element_degree = 1,
        time_stencil_size = 3,
        meshsize = nx)
    
    model.timestep_size.assign(Delta_t)
    
    model.smoothing.assign(s)
    
    model.solid_velocity_relaxation_factor.assign(tau)
    
    model.topwall_heatflux_switchtime =  topwall_heatflux_switchtime +  \
        2.*model.time_tolerance
    
    model.topwall_heatflux_postswitch = q
    
    model.save_smoothing_sequence = True
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "melt_octadecane/with_heatflux/switchtime" 
        + str(topwall_heatflux_switchtime)
        + "_tf" + str(endtime) + "/"
        + "q" + str(q) + "/"
        + "second_order_"
        + "nx" + str(nx) + "_Deltat" + str(Delta_t) 
        + "_s" + str(s) + "_tau" + str(tau) + "/" + "tf" + str(endtime) + "/")
    
    model.solutions, _, _ = model.run(
        endtime = endtime,
        plot = fempy.models.convection_coupled_phasechange.plot,
        report = True)
    
    liquid_area = fempy.models.convection_coupled_phasechange.postprocess(
        model)["liquid_area"]
    
    print("Liquid area = {0}".format(liquid_area))
    
    assert(abs(liquid_area - expected_liquid_area) < tolerance)


def test__regression__validate__freeze_water():
    
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
    spatial_dimensions = 2
    
    
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
        spatial_dimensions = spatial_dimensions,
        meshsize = nx)
    
    model.timestep_size = model.timestep_size.assign(t_f/float(nt))
    
    model.solid_velocity_relaxation_factor = \
        model.solid_velocity_relaxation_factor.assign(tau)
    
    model.smoothing = model.smoothing.assign(s)
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "freeze_water/" +
        "s{0}_tau{1}/rx{2}_nx{3}_rt{4}_nt{5}/q{6}/tf{7}/dim{8}/".format(
            s, tau, rx, nx, rt, nt, q, t_f, spatial_dimensions))
    
    model.solutions, _, _ = model.run(
        endtime = t_f,
        write_solution = False,
        plot = fempy.models.convection_coupled_phasechange.plot,
        report = True)
    
    liquid_area = fempy.models.convection_coupled_phasechange.postprocess(
        model)["liquid_area"]
    
    print("Liquid area = {0}".format(liquid_area))
    
    assert(abs(liquid_area - expected_liquid_area) < tolerance)
    