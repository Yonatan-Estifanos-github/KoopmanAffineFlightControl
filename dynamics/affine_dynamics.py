from numpy import dot
import numpy as np

from dynamics.dynamics import Dynamics

class AffineDynamics(Dynamics):
    """Abstract class for dynamics of the form x_dot = f(x, t) + g(x, t) * u.

    Override eval, drift, act.
    """

    def drift(self, x, t):
        """Compute drift vector f(x, t).

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Drift vector: numpy array
        """

        pass

    def act(self, x, t):
        """Compute actuation matrix g(x, t).

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Actuation matrix: numpy array
        """

        pass

    def eval_dot(self, x, u, t):
        """Evaluate the system dynamics at state x with input u at time t.
        Inputs:
        State, x: numpy array
        Control input, u: numpy array
        Time, t: float
        Outputs:
        State derivative x_dot: numpy array
        """
        
        #return self.drift(x, t) + dot(self.act(x, t), u)
        return self.drift(x, t) + np.dot(self.act(x, t), u)

