import firedrake as fe 
import sunfire.mms
import sunfire.benchmarks.melt_octadecane_in_cavity
import sunfire.benchmarks.freeze_water_in_cavity
import sunfire.test


datadir = sunfire.test.datadir

def test__validate__melt_octadecane__regression():
    
    tf = 80.
    
    th = 40.
    
    h = -0.02
    
    
    s = 1./200.
    
    tau = 1.e-12
    
    
    rx = 2
    
    nx = 24
    
    
    rt = 2
    
    nt = 4
    
    
    q = 4
    
    
    expected_liquid_area = 0.43
    
    tolerance = 0.01
    
    
    sim = sunfire.benchmarks.melt_octadecane_in_cavity.Simulation(
        quadrature_degree = q,
        element_degree = rx - 1,
        time_stencil_size = rt + 1,
        meshsize = nx,
        output_directory_path = "output/melt_octadecane/" 
            + "th{0}_h{1}/".format(th, h)
            + "s{0}_tau{1}/".format(s, tau)
            + "rx{0}_nx{1}_rt{2}_nt{3}/".format(rx, nx, rt, nt)
            + "q{0}/".format(q))
    
    
    sim.timestep_size.assign(tf/float(nt))
    
    sim.smoothing.assign(s)
    
    sim.solid_velocity_relaxation_factor.assign(tau)
    
    sim.topwall_heatflux_switchtime = th
    
    sim.topwall_heatflux_postswitch = h
    
    
    sim.solutions, _, = sim.run(
        endtime = tf,
        topwall_heatflux_poststart = h,
        topwall_heatflux_starttime = th)
    
    
    print("Liquid area = {0}".format(sim.liquid_area))
    
    assert(abs(sim.liquid_area - expected_liquid_area) < tolerance)
    

def freeze_water(endtime, s, tau, rx, nx, rt, nt, q, dim = 2, outdir = ""):
    
    mu_l__SI = 8.90e-4  # [Pa s]
    
    rho_l__SI = 999.84  # [kg / m^3]
    
    nu_l__SI = mu_l__SI/rho_l__SI  # [m^2 / s]
    
    t_f__SI = endtime  # [s]
    
    L__SI = 0.038  # [m]
    
    Tau = pow(L__SI, 2)/nu_l__SI
    
    t_f = t_f__SI/Tau
    
    """ For Kowalewski's water freezing experiment,
    at t_f__SI 2340 s, t_f = 1.44.
    """
    
    sim = sunfire.benchmarks.freeze_water_in_cavity.Simulation(
        quadrature_degree = q,
        element_degree = rx - 1,
        time_stencil_size = rt + 1,
        spatial_dimensions = dim,
        meshsize = nx,
        output_directory_path = str(outdir.join(
            "freeze_water/"
            + "dim{0}/".format(dim)
            + "s{0}_tau{1}/".format(s, tau)
            + "rx{0}_nx{1}_rt{2}_nt{3}/".format(rx, nx, rt, nt)
            + "q{0}/".format(q))))
    
    sim.timestep_size = sim.timestep_size.assign(t_f/float(nt))
    
    sim.solid_velocity_relaxation_factor = \
        sim.solid_velocity_relaxation_factor.assign(tau)
    
    sim.smoothing = sim.smoothing.assign(s)
    
    
    sim.solutions, _, = sim.run(endtime = t_f)
    
    
    print("Liquid area = {0}".format(sim.liquid_area))
    
    return sim
    
    
def test__validate__freeze_water__regression(datadir):
    
    sim = freeze_water(
        outdir = datadir,
        endtime = 2340.,
        s = 1./200.,
        tau = 1.e-12,
        rx = 2,
        nx = 24,
        rt = 2,
        nt = 4,
        q = 4)
    
    assert(abs(sim.liquid_area - 0.69) < 0.01)
    