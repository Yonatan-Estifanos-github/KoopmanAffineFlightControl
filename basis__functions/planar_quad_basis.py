import logging
from sklearn import preprocessing
import numpy as np
import itertools as it
from basis__functions.basis_functions import BasisFunctions

# Configure logging (ensure this is set up correctly according to your project structure)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PlanarQuadBasis(BasisFunctions):
    """Derived class for basis functions specifically for a planar quadrotor.
    
    Constructs polynomial and trigonometric basis functions to lift the state values.
    """

    def __init__(self, n, poly_deg=2, n_lift=None):
        """
        Initializes the PlanarQuadBasis with specified polynomial degree and dimensions.
        
        Parameters
        ----------
        n : int
            Number of original state dimensions
        poly_deg : int
            Degree of polynomial features
        n_lift : int, optional
            Number of lifting functions (automatically determined if not provided)
        """
        self.poly_deg = poly_deg
        logging.debug(f"Initializing PlanarQuadBasis with n={n}, poly_deg={poly_deg}")
        super(PlanarQuadBasis, self).__init__(n, n_lift)

    def construct_basis(self):
        """
        Constructs the combined polynomial and trigonometric basis functions.
        """
        logging.info("Constructing polynomial and trigonometric basis functions.")
        try:
            poly_features = preprocessing.PolynomialFeatures(degree=self.poly_deg)
            poly_features.fit(np.zeros((1, 2)))  # Fit to an array of zeros to initialize
            poly_func = lambda x: poly_features.transform(np.hstack((x[:, 2:3], x[:, 5:6])))

            sine_func = lambda x: np.concatenate((np.sin(x[:, 2:3]), np.cos(x[:, 2:3])), axis=1)

            self.basis = lambda x: np.hstack((np.ones((x.shape[0], 1)),
                                              x,
                                              self.basis_product_(x, poly_func, sine_func)))
            self.n_lift = 1 + self.n + poly_features.n_output_features_ * sine_func(np.zeros((1, self.n))).shape[1]
            logging.info("Basis functions successfully constructed.")
        except Exception as e:
            logging.error("Failed to construct basis functions", exc_info=True)

    def basis_product_(self, x, basis_1, basis_2):
        """
        Computes the product of two basis functions evaluations element-wise.

        Parameters
        ----------
        x : numpy array
            Input state vector
        basis_1 : function
            First basis function
        basis_2 : function
            Second basis function

        Returns
        -------
        numpy array
            Product of the two basis evaluations
        """
        logging.debug("Calculating basis product.")
        basis_1_eval = basis_1(x)
        basis_2_eval = basis_2(x)
        product = np.multiply(np.tile(basis_1_eval, (1, basis_2_eval.shape[1])),
                              np.repeat(basis_2_eval, basis_1_eval.shape[1], axis=1))
        logging.debug("Basis product calculated.")
        return product

# Additional functions and usage scenarios would be defined here.
