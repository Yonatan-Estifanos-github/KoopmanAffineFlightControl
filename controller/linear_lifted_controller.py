import logging
from numpy import dot
from controller.controller import Controller

# Configure logging if not already configured elsewhere
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class LinearLiftedController(Controller):
    """Class for linear policies using lifted state representation."""

    def __init__(self, affine_dynamics, K):
        """
        Create a LinearController object that calculates control action as u = -K * x, 
        where x is the state lifted to a higher-dimensional space.

        Inputs:
        affine_dynamics : AffineDynamics
            The dynamics model providing methods to evaluate lifted state.
        K : numpy array
            Gain matrix used in the linear control policy.
        """
        super().__init__(affine_dynamics)  # Use super() to call the base class constructor
        self.K = K
        logging.info("LinearLiftedController initialized with provided dynamics and gain matrix.")

    def eval(self, x, t):
        """
        Evaluate the control action for a given state and time.

        Inputs:
        x : numpy array
            Current state of the system.
        t : float
            Current time.

        Returns:
        numpy array
            Computed control action.
        """
        try:
            # Compute lifted state using dynamics eval_z method
            z = self.dynamics.eval_z(x, t)
            # Calculate control action using the gain matrix
            u = -dot(self.K, z)
            logging.debug(f"Evaluated control action at time {t}: {u}")
            return u
        except Exception as e:
            logging.error("Failed to evaluate control action", exc_info=True)
            raise
