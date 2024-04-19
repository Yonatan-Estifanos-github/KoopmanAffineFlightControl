import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Controller:
    """Abstract policy class for control.
    
    This class should be inherited by other classes that define specific control strategies.
    Methods `eval`, `process`, and `reset` need to be implemented or overridden as necessary.
    """

    def __init__(self, dynamics):
        """Create a Controller object.

        Inputs:
        dynamics : Dynamics
            A Dynamics object that defines the dynamics of the system being controlled.
        """
        self.dynamics = dynamics
        logging.info(f"Controller initialized with dynamics: {dynamics}")

    def eval(self, x, t):
        """Compute general representation of an action.

        Inputs:
        x : numpy array
            State of the system at current time.
        t : float
            Current time.

        Outputs:
        Action : object
            Computed action based on the system's state and time.
        """
        logging.debug(f"Evaluating control action at time {t} for state {x}")
        # This method should be overridden by subclasses.
        raise NotImplementedError("This method should be overridden by subclasses.")

    def process(self, u):
        """Transform general representation of an action to a numpy array.

        Inputs:
        u : object
            The action to be transformed.

        Outputs:
        numpy array
            The action represented as a numpy array.
        """
        logging.debug(f"Processing action {u}")
        return u

    def reset(self):
        """Reset any controller state."""
        logging.info("Resetting controller state.")
        # This method may need to be overridden by subclasses if there are specific reset actions needed.
