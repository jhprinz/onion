import numpy as np
import pyemma.msm as pymsm


class MSM(pymsm.MSM):
    """
    Simple class handling MSM objects
    """

    def ahash(self):
        """
        Return the group generalized inverse for the A matrix (1 - P)

        Returns
        -------
        numpy.ndarray, shape=(N, N)
            the GGInverse A^{\#}

        """
        tm = self.P
        size = self.nstates

        eq = self.stationary_distribution

        im = np.identity(size)
        wm = np.tile(eq, (size, 1))

        # A = 1 - P

        return np.linalg.inv(im - (tm - wm)) - wm

    def mfpt_matrix(self):
        """
        Compute the mean first-passage time matrix from a pyemma.MSM object

        Parameters
        ----------
        msm : :obj:`PyEmma.MSM`
            the MSM object to be analyzed
        """
        tm = self.P
        size = self.nstates

        eq = self.stationary_distribution

        im = np.identity(size)
        wm = np.tile(eq, (size, 1))

        # A = 1 - P

        ahash =  np.linalg.inv(im - (tm - wm)) - wm
        adg = np.diag(np.diagonal(ahash))

        dm = np.diag(1.0 / eq)
        jm = np.ones((size,size))
        mm = (np.dot(im - ahash + np.dot(jm, adg), dm))

        return mm.T

    def vfpt_matrix(self):
        """
        Compute the mean first-passage time matrix from a pyemma.MSM object

        Parameters
        ----------
        msm : :obj:`PyEmma.MSM`
            the MSM object to be analyzed
        """
        tm = self.P
        size = self.nstates

        eq = self.stationary_distribution

        im = np.identity(size)
        wm = np.tile(eq, (size, 1))

        # A = 1 - P

        ahash = np.linalg.inv(im - (tm - wm)) - wm
        adg = np.diag(np.diagonal(ahash))

        dm = np.diag(1.0 / eq)
        jm = np.ones((size, size))
        mm = (np.dot(im - ahash + np.dot(jm, adg), dm))

        ahm = np.dot(ahash, mm)
        ahmdg = np.diag(np.diagonal(ahm))

        bm = \
            np.dot(mm, 2.0 * np.dot(adg, dm) + im) + \
            2.0 * (ahm - np.dot(jm, ahmdg))

        vm = bm - np.square(mm)

        return vm.T
