#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 12:07:02 2018

@author: jan
"""
import numpy as np
from scipy.special import gamma


class _BaseOptim(object):

    def __init__(self, dim, n_particles, lb, ub, n_iters):
        self.params = {}
        self.params['dim'] = dim
        self.params['n_particles'] = n_particles
        self.params['lb'] = lb
        self.params['ub'] = ub
        self.params['n_iters'] = n_iters
        self._global_best = []
        self._state = 'virgin'

        self.fun = None

    def _log(self, global_best):
        self._global_best.append(global_best)

    def _get_solution_evol(self):
        if self._state == 'virgin':
            raise ValueError("Run the optimization first.")
        else:
            return self._global_best

    def get_solution(self):
        if self._state == 'virgin':
            raise ValueError("Run the optimization first.")
        else:
            return self._global_best[-1], self.fun(self._global_best[-1])


class ParticleSwarm(_BaseOptim):
    def __init__(self, obj_fun, dim, n_particles, lb, ub, n_iters, c1=2, c2=2,
                 w=.5, vmax=None):

        # pylint: disable=too-many-instance-attributes
        # pylint: disable=too-many-arguments

        """
        Parameters
        ----------
        obj_fun : function
            Objective function to minimize.

        dim : int
            Search space dimension.

        n_particles : int
            Number of particles in the swarm.

        lb : array-like
            Lower bounds for search space.

        ub : array-like
            Upper bounds for search space.

        n_iters : int
            Number of algorithm iterations.

        c1 : float
            Cognitive bias, favors particles best solution.

        c2 : float
            Social bias, favors swarm best solution.

        w : float
            Inertial weight, w>1 widens the swarm and w<1 vice-versa.

        vmax : float, optional
                Maximum velocity to move the particle, if introduced.
        """

        super(ParticleSwarm, self).__init__(dim=dim,
                                            n_particles=n_particles,
                                            lb=lb,
                                            ub=ub,
                                            n_iters=n_iters)
        self.fun = obj_fun
        self.params['c1'] = c1
        self.params['c2'] = c2
        self.params['w'] = w
        self.params['vmax'] = vmax

        self._particles = np.random.uniform(lb, ub, size=(n_particles, dim))
        self._v = np.zeros(shape=(n_particles, dim))
        self._particles = np.clip(self._particles, lb, ub)
        self._feval = np.array([obj_fun(prtcl) for prtcl in self._particles])
        self._p = self._particles  # particle best
        self._g = self._particles[self._feval.argmin(), :]  # global best

    def run(self):
        t = 0
        while t < self.params['n_iters']:
            t += 1

            self._update_particles()
            fp = np.array([self.fun(prtcl) for prtcl in self._particles])
            best = fp.min()
            best_idx = fp.argmin()

            if best < self.fun(self._g):
                self._g = self._particles[best_idx]

            idx = self._feval > fp
            self._feval[idx] = fp[idx]
            self._p[idx, :] = self._particles[idx, :]

            self._log(self._g)

        self._state = 'optim'

    def _update_particles(self):
        n_particles = self.params['n_particles']
        dim = self.params['dim']
        lb = self.params['lb']
        ub = self.params['ub']
        c1 = self.params['c1']
        c2 = self.params['c2']
        w = self.params['w']
        vmax = self.params['vmax']

        eta1, eta2 = (np.random.uniform(size=(n_particles, dim)),
                      np.random.uniform(size=(n_particles, dim)))

        self._v = (w * self._v +
                   c1 * eta1 * (self._p - self._particles) +
                   c2 * eta2 * (self._g - self._particles))

        # TODO:  smarter indexing of self._v <> vmax
        if vmax is not None:
            for i in range(self._v.shape[0]):
                idx_greater = self._v[i, :] > vmax
                idx_smaller = self._v[i, :] < -vmax
                self._v[i, idx_greater] = vmax[idx_greater]
                self._v[i, idx_smaller] = -vmax[idx_smaller]

        self._particles += self._v
        self._particles = np.clip(self._particles, lb, ub)


class CuckooSearch(_BaseOptim):
    """Cuckoo Search implementation."""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments

    def __init__(self, obj_fun, dim, n_particles, n_nests, lb, ub, n_iters,
                 pa=.25):
        """
        Parameters
        ----------
        obj_fun : function
            Objective function to minimize.

        dim : int
            Search space dimension.

        n_cuckoos : int
            Number of cuckoos.

        n_nests : int
            Number of nests.

        lb : array-like
            Lower bounds for search space.

        ub : array-like
            Upper bounds for search space.

        n_iters : int
            Number of algorithm iterations.

        pa : float
            Probability of nests re-initialization.
        """

        super(CuckooSearch, self).__init__(dim=dim,
                                           n_particles=n_particles,
                                           lb=lb,
                                           ub=ub,
                                           n_iters=n_iters)

        self.fun = obj_fun
        self.params['n_nests'] = n_nests
        self.params['pa'] = pa

        self._nests = np.random.uniform(lb, ub, size=(n_nests, dim))
        self._particles = np.random.uniform(lb, ub, size=(n_particles, dim))

        self._gbest = self._nests[np.array([self.fun(nest)
                                            for nest in self._nests]).argmin()]

        self._step = self.__step(dim)

    def run(self):
        n_nests = self.params['n_nests']
        n_iters = self.params['n_iters']
        n_particles = self.params['n_particles']
        pa = self.params['pa']
        t = 0
        while t < n_iters:
            t += 1

            for cuckoo in self._particles:
                idx = np.random.randint(0, n_nests-1)
                if self.fun(cuckoo) < self.fun(self._nests[idx]):
                    self._nests[idx] = cuckoo

            nests_res = [(self.fun(self._nests[i]), i)
                         for i in range(n_nests)]
            nests_res.sort()

            cuckoos_res = [(self.fun(self._particles[i]), i)
                           for i in range(n_particles)]
            cuckoos_res.sort(reverse=True)

            idx_worsts = [nests_res[-i - 1][1]
                          for i in range(n_nests // 2)]
            for i in idx_worsts:
                if np.random.uniform() < pa:
                    self._nests[-i-1] = self._init_nest()

            n = n_particles if n_nests > n_particles else n_nests
            for idx in range(n):
                if nests_res[idx][0] < cuckoos_res[idx][0]:
                    self._particles[idx] = self._nests[idx]

            self._new_flight()

            # current best solution vs proposed
            best = self._nests[np.array([self.fun(nest)
                                        for nest in self._nests]).argmin()]

            if self.fun(self._gbest) > self.fun(best):
                self._gbest = best

            self._log(self._gbest)

        self._state = 'optim'

    @staticmethod
    def __step(dim):
        beta = 3 / 2
        sigma = (gamma(1+beta) * np.sin(np.pi*beta/2) / (
                 gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
        u = np.random.normal(size=(dim)) * sigma
        v = np.random.normal(size=(dim))
        return u / abs(v)**(1 / beta)

    def _init_nest(self):
        lb = self.params['lb']
        ub = self.params['ub']
        dim = self.params['dim']
        return np.random.uniform(lb, ub, size=dim)

    def _new_flight(self):
        """Generates Levy distributed random variable.
        """
        dim = self.params['dim']
        alpha = .2 * self._step * (self._particles - self._gbest)
        self._particles += alpha * np.random.normal(loc=1, scale=1,
                                                    size=dim)
        self._constrain_space()

    def _constrain_space(self):
        lb = self.params['lb']
        ub = self.params['ub']
        self._nests = np.clip(self._nests, lb, ub)
        self._particles = np.clip(self._particles, lb, ub)
