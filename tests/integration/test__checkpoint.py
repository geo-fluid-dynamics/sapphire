"""Tests for checkpoint/restart features"""
import matplotlib
matplotlib.use('Agg')  # Only use back-end to prevent displaying image
import matplotlib.pyplot
import firedrake as fe 
import sapphire.simulations.examples.heat_driven_cavity
import sapphire.output


def test__plot_from_checkpoint(tmpdir):
    """Plot solution loaded from a checkpoint."""
    sim = sapphire.simulations.examples.heat_driven_cavity.Simulation(
        taylor_hood_pressure_degree = 1,
        temperature_degree = 2,
        mesh_dimensions=(40, 40),
        output_directory_path=tmpdir)
    
    
    sim.solution = sim.solve()
    
    sim.write_checkpoint()
    
    
    solution = fe.Function(
        function_space=sim.solution.function_space(),
        name=sim.solution.name())
    
    states = [{"solution": solution,
               "time": sim.state["time"],
               "index": sim.state["index"]},]
    
    states = sapphire.output.read_checkpoint(
        states=states,
        dirpath=tmpdir,
        filename="checkpoints")
    
    p, u, T = solution.split()
    
    fe.tricontourf(T)
    
    filepath = tmpdir+"/T.png"
    
    print("Writing plot to {}".format(filepath))
    
    matplotlib.pyplot.savefig(filepath)
    