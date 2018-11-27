import firedrake as fe 
import fempy.mms
import fempy.models.convection_coupled_phasechange
import fempy.benchmarks.melting_octadecane


def test__melting_octadecane_benchmark():

    model = fempy.benchmarks.melting_octadecane.Model(meshsize = 40)
    
    model.run(endtime = 80.)
