import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from controller.controller import Controller
import numpy as np
import logging

class ConstantController(Controller):
    """Class for constant action policies."""

    def __init__(self, dynamics, u_const):
        """Create a ConstantController object.

        Inputs:
        Dynamics, dynamics: Dynamics object managing the dynamics of the system
        Constant action, u_const: numpy array defining the constant control action
        """
        super().__init__(dynamics)
        self.u_const = u_const
        logging.info(f"ConstantController initialized with constant action: {u_const}")

    def eval(self, x, t):
        """Evaluate the control action.

        Parameters:
        x : numpy array
            The current state (not used in this controller).
        t : float
            The current time (not used in this controller).

        Returns:
        numpy array
            The constant control action.
        """
        logging.debug(f"Constant control action returned: {self.u_const} at time {t}")
        return self.u_const
