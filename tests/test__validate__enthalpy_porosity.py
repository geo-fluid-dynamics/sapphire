import firedrake as fe 
import fempy.mms
import fempy.benchmarks.melting_octadecane


def test__regression__validate__melting_octadecane_with_heat_flux():
    
    endtime = 80.
    
    topwall_heatflux_switchtime = 40.
    
    q = -0.02
    
    
    s = 1./64.
    
    nx = 32
    
    Delta_t = 10.
    
    tau = 1.e-12
    
    
    expected_liquid_area = 0.64
    
    tolerance = 0.01
    
    
    model = fempy.benchmarks.melting_octadecane.Model(
        quadrature_degree = 4,
        spatial_order = 2,
        temporal_order = 2,
        meshsize = nx)
    
    model.timestep_size.assign(Delta_t)
    
    model.smoothing.assign(s)
    
    model.solid_velocity_relaxation_factor.assign(tau)
    
    model.topwall_heatflux_switchtime =  topwall_heatflux_switchtime +  \
        2.*model.time_tolerance
    
    model.topwall_heatflux_postswitch = q
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "melting_octadecane/with_heatflux/switchtime" 
        + str(topwall_heatflux_switchtime)
        + "_tf" + str(endtime) + "/"
        + "q" + str(q) + "/"
        + "second_order_"
        + "nx" + str(nx) + "_Deltat" + str(Delta_t) 
        + "_s" + str(s) + "_tau" + str(tau) + "/" + "tf" + str(endtime) + "/")
        
    model.run(endtime = endtime, plot = False, report = False)
    
        
    p, u, T = model.solution.split()
    
    phil = model.porosity(T)
    
    liquid_area = fe.assemble(phil*fe.dx)
    
    print("Liquid area = " + str(liquid_area))
    
    assert(abs(liquid_area - expected_liquid_area) < tolerance)


def test__regression__validate__melting_octadecane_without_heat_flux():
    
    endtime, expected_liquid_area, tolerance = 30., 0.22, 0.01
    
    nx = 32
    
    Delta_t = 10.
    
    tau = 1.e-12
    
    model = fempy.benchmarks.melting_octadecane.Model(
        quadrature_degree = 4,
        spatial_order = 2,
        temporal_order = 2,
        meshsize = nx)
    
    model.topwall_heatflux.assign(0.)
    
    model.timestep_size.assign(Delta_t)
    
    s = 1./256.
    
    model.smoothing.assign(s)
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "melting_octadecane/without_heat_flux/second_order/" 
        + "nx" + str(nx) + "_Deltat" + str(Delta_t) 
        + "_s" + str(s) + "_tau" + str(tau) + "/tf" + str(endtime) + "/")
        
    model.run(endtime = endtime, plot = False, report = False)
    
    p, u, T = model.solution.split()
    
    phil = model.porosity(T)
    
    liquid_area = fe.assemble(phil*fe.dx)
    
    print("Liquid area = " + str(liquid_area))
    
    assert(abs(liquid_area - expected_liquid_area) < tolerance)
    
    phil_h = fe.interpolate(phil, model.function_space.sub(2))
    
    max_phil = phil_h.vector().max()
    
    print("Maximum phil = " + str(max_phil))
    
    assert(abs(max_phil - 1.) < tolerance)
    