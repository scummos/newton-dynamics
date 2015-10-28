#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np

class ParticleSystem:
    def __init__(self, x0, v0, m, dt, t_end):
        assert len(m) == x0.shape[0]
        assert v0.shape == x0.shape
        assert dt < t_end

        ndim = 3
        self.t = np.arange(0, t_end, dt)
        self.xt = np.zeros((self.t.shape[0], x0.shape[0], ndim))
        self.vt = np.zeros_like(self.xt)

        self.xt[0] = x0
        self.vt[0] = v0

        self.g = 9.81
        self.m = m
        self.dt = dt

    def kinetic_energy(self):
        ekin = np.sum(np.sum(0.5 * self.m * self.vt**2, axis=2), axis=1)
        assert ekin.shape == self.t.shape
        return ekin

    def potential_energy(self):
        raise NotImplementedError()

    def energy(self):
        return self.kinetic_energy() + self.potential_energy()

    def force(self, t_index):
        raise NotImplementedError()

class GravityParticleSystem(ParticleSystem):
    def force(self, t_index):
        return np.array([0, 0, -self.m * self.g])

    def potential_energy(self):
        epot = np.sum(self.m * self.g * self.xt[:,:,2], axis=1)
        assert epot.shape == self.t.shape
        return epot

class VelocityVerletPropagator:
    def __init__(self, system: ParticleSystem):
        self.system = system

    def run(self):
        for index, t in enumerate(self.system.t[:-1]):
            self.step(index)

    def step(self, t_index):
        dt = self.system.dt
        dtsq = dt*dt
        ri = self.system.xt[t_index]
        vi = self.system.vt[t_index]

        fi = self.system.force(t_index)
        xnext = ri + vi*dt + fi/(2*self.system.m)*dtsq
        self.system.xt[t_index + 1] = xnext

        fnext = self.system.force(t_index + 1)
        vnext = vi + (fi+fnext)/(2*self.system.m)*dt
        self.system.vt[t_index + 1] = vnext

if __name__ == '__main__':
    x0 = np.array([[0, 0, 0]])
    v0 = np.array([[15, 15, 30]])
    m = np.array([1])
    dt = 0.0025
    t_end = 10
    sys = GravityParticleSystem(x0, v0, m, dt, t_end)
    propagator = VelocityVerletPropagator(sys)
    propagator.run()

    #plt.plot(sys.t, sys.xt[:,0,2])
    #plt.plot(sys.t, sys.vt[:,0,2])
    energy = sys.energy()
    plt.plot(sys.t, energy / energy[0])
    plt.grid()
    plt.show()


