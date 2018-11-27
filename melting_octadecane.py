from fempy.benchmarks.melting_octadecane import Model
import firedrake as fe
import matplotlib.pyplot as plt


model = Model(meshsize = 40)

#model.liquidus_temperature.assign(0.)

#s = 0.01

#model.phase_interface_smoothing.assign(s)

#model.smoothing_sequence = (64.*s, 32.*s, 16.*s, 8.*s, 4.*s, 2.*s, s)

model.run(endtime = 70.)

p, u, T = model.solution.split()

plt.figure()

fe.plot(u)

plt.xlabel(r"$x$")

plt.ylabel(r"$y$")

plt.title("$u$")

plt.savefig("u_t" + str(model.time.__float__()) + ".png")
