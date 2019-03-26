import firedrake as fe 
import fempy.mms
import fempy.benchmarks.melt_octadecane_in_cavity
import fempy.benchmarks.freeze_water_in_cavity
import fempy.test


datadir = fempy.test.datadir

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
    
    
    model = fempy.benchmarks.melt_octadecane_in_cavity.Model(
        quadrature_degree = q,
        element_degree = rx - 1,
        time_stencil_size = rt + 1,
        meshsize = nx,
        output_directory_path = "output/melt_octadecane/" 
            + "th{0}_h{1}/".format(th, h)
            + "s{0}_tau{1}/".format(s, tau)
            + "rx{0}_nx{1}_rt{2}_nt{3}/".format(rx, nx, rt, nt)
            + "q{0}/".format(q))
    
    
    model.timestep_size.assign(tf/float(nt))
    
    model.smoothing.assign(s)
    
    model.solid_velocity_relaxation_factor.assign(tau)
    
    model.topwall_heatflux_switchtime = th
    
    model.topwall_heatflux_postswitch = h
    
    
    model.solutions, _, = model.run(
        endtime = tf,
        topwall_heatflux_poststart = h,
        topwall_heatflux_starttime = th)
    
    
    print("Liquid area = {0}".format(model.liquid_area))
    
    assert(abs(model.liquid_area - expected_liquid_area) < tolerance)
    

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
    
    model = fempy.benchmarks.freeze_water_in_cavity.Model(
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
    
    model.timestep_size = model.timestep_size.assign(t_f/float(nt))
    
    model.solid_velocity_relaxation_factor = \
        model.solid_velocity_relaxation_factor.assign(tau)
    
    model.smoothing = model.smoothing.assign(s)
    
    
    model.solutions, _, = model.run(endtime = t_f)
    
    
    print("Liquid area = {0}".format(model.liquid_area))
    
    return model
    
    
def test__validate__freeze_water__regression(datadir):
    
    model = freeze_water(
        outdir = datadir,
        endtime = 2340.,
        s = 1./200.,
        tau = 1.e-12,
        rx = 2,
        nx = 24,
        rt = 2,
        nt = 4,
        q = 4)
    
    assert(abs(model.liquid_area - 0.69) < 0.01)
    
    
def test__freeze_water__parameter_dependence__long(datadir):
    
    s0 = 1./200.
    
    tau0 = 1.e-12
    
    nx0 = 24
    
    nt0 = 5
    
    q0 = 8
    
    for s, tau, nx, nt, q in (
            (s0, tau0, nx0, nt0, q0),
            (1./400., tau0, nx0, nt0, q0),
            (s0, 1.e-6, nx0, nt0, q0),
            (s0, tau0, 36, nt0, q0),
            (s0, tau0, nx0, 10, q0),
            (s0, tau0, nx0, nt0, 4)):
        
        try:
        
            freeze_water(
                outdir = datadir,
                endtime = 2600.,
                s = s,
                tau = tau,
                rx = 2,
                nx = nx,
                rt = 2,
                nt = nt,
                q = q)
            
        except fe.exceptions.ConvergenceError:
        
            continue
            