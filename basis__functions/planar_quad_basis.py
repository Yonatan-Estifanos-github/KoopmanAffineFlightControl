from sklearn import preprocessing
import numpy as np
import itertools as it
from basis__functions.basis_functions import BasisFunctions


class PlanarQuadBasis(BasisFunctions):
    """Abstract class for basis functions that can be used to "lift" the state values.
    
    Override construct_basis
    """

    def __init__(self, n, poly_deg=2, n_lift=None):
        """
        Parameters
        ----------
        n : int
            number of basis functions
        Nlift : int
            Number of lifing functions
        """
        self.poly_deg = poly_deg
        super(PlanarQuadBasis, self).__init__(n, n_lift)

    def construct_basis(self):
        poly_features = preprocessing.PolynomialFeatures(degree=self.poly_deg)
        poly_features.fit(np.zeros((1, 2)))
        poly_func = lambda x: poly_features.transform(np.hstack((x[:,2:3], x[:,5:6])))
        sine_func = lambda x: np.concatenate((np.sin(x[:,2:3]), np.cos(x[:,2:3])), axis=1)

        self.basis = lambda x: np.hstack((np.ones((x.shape[0],1)),
                                          x,
                                          self.basis_product_(x, poly_func, sine_func)))
        self.n_lift = 1 + self.n + poly_features.n_output_features_*sine_func(np.zeros((1,self.n))).shape[1]

    def basis_product_(self, x, basis_1, basis_2):
        basis_1_eval = basis_1(x)
        basis_2_eval = basis_2(x)

        return np.multiply(np.tile(basis_1_eval, (1,basis_2_eval.shape[1])), np.repeat(basis_2_eval, basis_1_eval.shape[1], axis=1))
