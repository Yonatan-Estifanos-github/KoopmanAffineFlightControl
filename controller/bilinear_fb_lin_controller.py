from numpy import dot, concatenate, zeros
from numpy.linalg import solve

from controller.controller import Controller

import logging

# Configuring logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


import numpy as np
from numpy.linalg import solve
from controller.controller import Controller

class BilinearFBLinController(Controller):
    """Class for bilinear linearizing feedback policies."""

    def __init__(self, bilinear_dynamics, output, lin_controller_gain):
        """Create an FBLinController object.
        
        Policy is u = (act)^-1 * (aux), where drift and act are
        components of drift vector and actuation matrix corresponding to
        highest-order derivatives of each output coordinate and aux is an
        auxilliary linear controller.

        Inputs:
        Bilinear dynamics, bilinear_dynamics: FBLinDynamics
        Output, output: Output
        Auxilliary linear controller, lin_controller_gain: array
        """
        super().__init__(bilinear_dynamics)
        self.dynamics = bilinear_dynamics
        self.output = output
        self.lin_controller_gain = lin_controller_gain
        self.u_prev = np.zeros(self.dynamics.m)

        logging.info("BilinearFBLinController initialized with given dynamics and linear controller gain settings.")

    def eval(self, x, t):
        """Evaluate the control action based on the state x and time t."""
        try:
            z = self.dynamics.basis(x.reshape((1, -1))).squeeze()
            z_dot = self.dynamics.eval_dot(z, self.u_prev, t)
            eta_z = np.concatenate((z - self.output.z_d(t), z_dot - self.output.z_d_dot(t)))
            nu = -np.dot(self.lin_controller_gain, eta_z)

            act = self.dynamics.act(z, t)
            F = self.dynamics.A

            u = solve(self.output.C_h @ F @ act, self.output.C_h @ (self.output.z_d_ddot(t) - F @ F @ self.output.z_d(t) + nu))
            self.u_prev = u

            logging.debug(f"Control action calculated at time {t}: {u}")
            return u
        except Exception as e:
            logging.error("Failed to evaluate the control action", exc_info=True)
            raise



