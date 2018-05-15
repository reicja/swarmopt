#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 19:18:12 2018

@author: jan
"""
# 2D Rosenbrock function
# (x, y) -> a(1-x^2) + b(y-x^2)^2, for a, b = 1, 100
# with global minimum (1, 1) -> 0.

from swarm_optimization import CuckooSearch, ParticleSwarm

testfun = lambda x: (1-x[0])**2 + 100*(x[1] - x[0]**2)**2
dim = 2
n_particles = 50
lb, ub = -10, 10
vmax = [5, 5]
n_iters = 1000

cs = CuckooSearch(testfun,
                  dim=dim,
                  n_particles=n_particles,
                  n_nests=100,
                  lb=lb,
                  ub=ub,
                  n_iters=n_iters)

cs.run()
print "Cuckoo Search solution {0}".format(cs.get_solution())

ps = ParticleSwarm(testfun,
                   dim=dim,
                   n_particles=n_particles,
                   lb=lb,
                   ub=ub,
                   n_iters=n_iters,
                   vmax=vmax,
                   c1=0.05,
                   c2=3.4,
                   w=.4)
ps.run()
print "Particle Swarm solution {0}".format(ps.get_solution())
