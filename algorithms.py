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

    def kinetic_energy(self) -> np.ndarray:
        ekin = np.sum(np.sum(0.5 * self.m * self.vt**2, axis=2), axis=1)
        assert ekin.shape == self.t.shape
        return ekin

    def potential_energy(self) -> np.ndarray:
        raise NotImplementedError()

    def energy(self) -> np.ndarray:
        return self.kinetic_energy() + self.potential_energy()

    def force(self, t_index):
        raise NotImplementedError()


class ConstantGravityParticleSystem(ParticleSystem):
    def force(self, t, x, v):
        return np.array([0, 0, -self.m * self.g])*np.ones(self.xt.shape[1:])

    def potential_energy(self):
        epot = np.sum(self.m * self.g * self.xt[:,:,2], axis=1)
        assert epot.shape == self.t.shape
        return epot


class AerodynamicParticleSystem(ConstantGravityParticleSystem):
    def force(self, t, x, v):
        rho = 1.2
        cw = 0.45
        A = 100e-4
        fdiss = -rho*cw*A * np.abs(v)**3*v / 2
        fg = super(AerodynamicParticleSystem, self).force(t, x, v)
        return fg + fdiss


class NewtonPropagator:
    def __init__(self, system: ParticleSystem):
        self.system = system

    def run(self):
        print("running {0} steps".format(len(self.system.t) - 1))
        for index, t in enumerate(self.system.t[:-1]):
            self.step(index)

    def step(self):
        raise NotImplementedError()

class VelocityVerletPropagator(NewtonPropagator):
    def step(self, t_index):
        dt = self.system.dt
        dtsq = dt*dt
        ri = self.system.xt[t_index]
        vi = self.system.vt[t_index]
        m = self.system.m

        fi = self.system.force(t_index*dt, ri, None)
        rnext = ri + vi*dt + fi/(2*m)*dtsq
        self.system.xt[t_index + 1] = rnext

        fnext = self.system.force(t_index*dt + dt, rnext, None)
        vnext = vi + (fi+fnext)/(2*m)*dt
        self.system.vt[t_index + 1] = vnext


class Rk4Propagator(NewtonPropagator):
    def step(self, t_index):
        dt = self.system.dt
        t = t_index * dt
        y = np.array([self.system.xt[t_index], self.system.vt[t_index]])
        def f(t, y):
            return np.array([y[1], self.system.force(t, *y)/self.system.m])

        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2*k1)
        k3 = f(t + dt/2, y + dt/2*k2)
        k4 = f(t + dt,   y + dt * k3)

        yn = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        self.system.xt[t_index + 1], self.system.vt[t_index + 1] = yn


if __name__ == '__main__':
    x0 = np.array([[0, 0, 0]])
    v0 = np.array([[15, 15, 30]])
    m = np.array([1])
    dt = 0.0025
    t_end = 5

    sys = AerodynamicParticleSystem(x0, v0, m, dt, t_end)
    propagator = Rk4Propagator(sys)
    propagator.run()

    plt.plot(sys.t, sys.xt[:,0,2], label="z[t]")
    plt.plot(sys.t, sys.vt[:,0,1], label="vy[t]")
    plt.plot(sys.t, sys.vt[:,0,2], label="vz[t]")
    plt.legend()

    #energy = sys.energy()
    #plt.plot(sys.t, energy / energy[0])

    plt.grid()
    plt.show()
