import logging
import os


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class BasisFunctions():
    """Abstract class for basis functions that can be used to "lift" the state values.
    
    Override construct_basis
    """

    def __init__(self, n, n_lift):
        """
        Parameters
        ----------
        n : int
            number of basis functions
        n_lift : int
            Number of lifting functions
        """
        self.n = n
        self.n_lift = n_lift
        self.basis = None
        logging.info(f"BasisFunctions initialized with {n} basis functions and {n_lift} lifting functions.")

    def lift(self, q):
        """
        Call this function to get the variables in lifted space

        Parameters
        ----------
        q : numpy array
            State vector

        Returns
        -------
        basis applied to q
        """
        try:
            result = self.basis(q)
            logging.debug(f"Lift operation successful on state vector: {q}")
            return result
        except Exception as e:
            logging.error(f"Failed to apply basis to the state vector {q}: {e}")
            raise

    def construct_basis(self):
        """
        Construct the basis functions. This method needs to be overridden by subclasses.
        """
        logging.info("construct_basis method needs to be overridden in subclasses.")
        pass

