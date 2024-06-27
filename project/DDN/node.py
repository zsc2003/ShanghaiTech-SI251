# DEEP DECLARATIVE NODES

import autograd.numpy as np
import scipy as sci
from autograd import grad, jacobian
import warnings

class AbstractNode:

    def __init__(self, dim_x=1, dim_y=1):
        assert (dim_x > 0) and (dim_y > 0)
        self.dim_x = dim_x # dimensionality of input variable
        self.dim_y = dim_y # dimensionality of output variable

    def solve(self, x):
        raise NotImplementedError()

    def gradient(self, x, y=None, ctx=None):
        raise NotImplementedError()


class AbstractDeclarativeNode(AbstractNode):
    #eps = 1.0e-6 # tolerance for checking that optimality conditions are satisfied

    def __init__(self, dim_x=1, dim_y=1):
        super().__init__(dim_x, dim_y)
        self.fY = grad(self.objective, 1)
        self.fYY = jacobian(self.fY, 1)
        self.fXY = jacobian(self.fY, 0)
        self.eps = 1.0e-6

    def objective(self, x, y):
        warnings.warn("objective function not implemented.")
        result = 0.0
        return result

    def solve(self, x):
        raise NotImplementedError()

    def gradient(self, x, y=None, ctx=None):
        if y is None:
            y, ctx = self.solve(x)

        if not self._check_optimality_cond(x, y):
            return None

        # assert self.check_optimality_cond(x, y)
        return -1.0 * np.matmul(np.linalg.inv(self.fYY(x, y)),self.fXY(x, y))

    # def check_optimality_cond(self, x, y, ctx=None):
    #     return all((abs(self.fY(x, y)) <= self.eps))

    def _check_optimality_cond(self, x, y, ctx=None):
        # Initialize a flag to check if all elements meet the condition
        all_within_eps = True

        # Iterate over each element and check the condition
        for i in range(len(self.fY(x, y))):
            if abs(self.fY(x, y)[i]) > self.eps:
                all_within_eps = False
                break

        return all_within_eps


class EqConstDeclarativeNode(AbstractDeclarativeNode):

    def __init__(self, dim_x, dim_y):
        super().__init__(dim_x, dim_y)
        self.hY = grad(self.constraint, 1)
        self.hX = grad(self.constraint, 0)
        self.hYY = jacobian(self.hY, 1)
        self.hXY = jacobian(self.hY, 0)

    def constraint(self, x, y):
        warnings.warn("constraint function not implemented.")
        return 0.0

    def solve(self, x):
        raise NotImplementedError()

    def gradient(self, x, y=None, ctx=None):

        # compute optimal value if not already done so
        if y is None:
            y, ctx = self.solve(x)
        assert self._check_constraints(x, y), [x, y, abs(self.constraint(x, y))]
        assert self._check_optimality_cond(x, y, ctx), [x, y, ctx]

        # nu = self._get_nu_star(x, y) if (ctx is None) else ctx['nu']
        if ctx is None:
            nu = self._get_nu_star(x, y)
        else:
            nu = ctx['nu']

        # return unconstrained gradient if nu is undefined
        if np.isnan(nu):
            # return -1.0 * np.linalg.solve(self.fYY(x, y), self.fXY(x, y))
            return -1.0 * np.matmul(np.linalg.inv(self.fYY(x, y)),self.fXY(x, y))

        H = self.fYY(x, y) - nu * self.hYY(x, y)
        a = self.hY(x, y)
        B = self.fXY(x, y) - nu * self.hXY(x, y)
        C = self.hX(x, y)
        try:
            v = sci.linalg.solve(H, np.concatenate((a.reshape((self.dim_y, 1)), B), axis=1), assume_a='pos')
        except:
            return np.full((self.dim_y, self.dim_x), np.nan).squeeze()
        assert v[:, 0].dot(a) != 0.0, "a^T H^{-1} a is zero"
        return (np.outer(v[:, 0], (v[:, 0].dot(B) - C) / v[:, 0].dot(a)) - v[:, 1:self.dim_x + 1]).squeeze()

    def _get_nu_star(self, x, y):
        indx = np.nonzero(self.hY(x, y))
        flag = True
        for i in range(len(self.hY(x, y))):
            if self.hY(x, y)[i] != 0:
                flag = False
                break

        if flag:
            return 0.0
        return self.fY(x, y)[indx[0][0]] / self.hY(x, y)[indx[0][0]]

    def _check_constraints(self, x, y):
        return abs(self.constraint(x, y)) <= self.eps

    def _check_optimality_cond(self, x, y, ctx=None):
        # nu = self._get_nu_star(x, y) if (ctx is None) else ctx['nu']
        if ctx is None:
            nu = self._get_nu_star(x, y)
        else:
            nu = ctx['nu']

        if np.isnan(nu):
            all_within_eps = True
            # Iterate over each element and check the condition
            for i in range(len(self.fY(x, y))):
                if abs(self.fY(x, y)[i]) > self.eps:
                    all_within_eps = False
                    break

            return all_within_eps

        # check for invalid lagrangian (gradient of constraint zero at optimal point)
        all_within_eps = True
        # Iterate over each element and check the condition
        for i in range(len(self.hY(x, y))):
            if abs(self.hY(x, y)[i]) > self.eps:
                all_within_eps = False
                break
        if all_within_eps:
            warnings.warn("gradient of constraint function vanishes at the optimum.")
            return True

        all_within_eps = True
        # Iterate over each element and check the condition
        eq = self.fY(x, y) - nu * self.hY(x, y)
        for i in range(len(eq)):
            if abs(eq[i]) > self.eps:
                all_within_eps = False
                break
        return all_within_eps
        # return (abs(self.fY(x, y) - nu * self.hY(x, y)) <= self.eps).all()


class IneqConstDeclarativeNode(EqConstDeclarativeNode):

    def __init__(self, dim_x, dim_y):
        super().__init__(dim_x, dim_y)

    def _get_nu_star(self, x, y):
        # if np.all(np.abs(self.fY(x, y)) < self.eps):
        #     return np.nan # flag that unconstrained gradient should be used
        all_within_eps = True
        # Iterate over each element and check the condition
        for i in range(len(self.fY(x, y))):
            if abs(self.fY(x, y)[i]) > self.eps:
                all_within_eps = False
                break

        if all_within_eps:
            return np.nan # flag that unconstrained gradient should be used

        indx = np.nonzero(self.hY(x, y))
        if len(indx[0]) == 0:
            return 0.0 # still use constrained gradient
        return self.fY(x, y)[indx[0][0]] / self.hY(x, y)[indx[0][0]]

    def _check_constraints(self, x, y):
        if self.constraint(x, y) <= self.eps:
            return True
        return False
        # return self.constraint(x, y) <= self.eps


class MultiEqConstDeclarativeNode(AbstractDeclarativeNode):

    def __init__(self, dim_x, dim_y):
        super().__init__(dim_x, dim_y)

        # partial derivatives of constraint function
        self.hY = jacobian(self.constraint, 1)
        self.hX = jacobian(self.constraint, 0)
        self.hYY = jacobian(self.hY, 1)
        self.hXY = jacobian(self.hY, 0)

    def constraint(self, x, y):
        warnings.warn("constraint function not implemented.")
        return 0.0

    def gradient(self, x, y=None, ctx=None):

        # compute optimal value if not already done so
        if y is None:
            y, ctx = self.solve(x)
            assert self._check_constraints(x, y)
            assert self._check_optimality_cond(x, y, ctx)

        if (ctx is None or 'nu' not in ctx):
            nu = self._get_nu_star(x, y)
        else:
            nu = ctx['nu']
        # nu = self._get_nu_star(x, y) if (ctx is None or 'nu' not in ctx) else ctx['nu']

        p = len(self.hY(x, y))

        H_temp = self.fYY(x, y) - np.sum(nu[i] * self.hYY(x, y)[i, :, :] for i in range(p))  # m-by-m
        H = (H_temp + H_temp.T) / 2   # make sure H is symmetric

        A = self.hY(x, y)   # p-by-m
        B = self.fXY(x, y) - np.sum(nu[i] * self.hXY(x, y)[i, :, :] for i in range(p))  # m-by-n
        C = self.hX(x, y)   # p-by-n

        # try to use cholesky to solve H^{-1}A^T and H^-1 B
        try:
            CC, L = sci.linalg.cho_factor(H)
            invHAT = sci.linalg.cho_solve((CC, L), A.T)
            invHB = sci.linalg.cho_solve((CC, L), B)
        # if H is not positive definite, revert to LU to solve
        except:
            invHAT = sci.linalg.solve(H, A.T)
            invHB = sci.linalg.solve(H, B)

        # compute Dy(x) = H^{-1}A^T(AH^{-1}A^T)^{-1}(AH^{-1}B-C) - H^{-1}B
        return np.dot(invHAT, sci.linalg.solve(np.dot(A, invHAT), np.dot(A, invHB) - C)) - invHB

    def _get_nu_star(self, x, y):
        nu = sci.linalg.lstsq(self.hY(x, y).T, self.fY(x, y))[0]
        return nu

    def _check_constraints(self, x, y):
        all_within_eps = True
        # Iterate over each element and check the condition
        for i in range(len(self.constraint(x, y))):
            if abs(self.constraint(x, y)[i]) > self.eps:
                all_within_eps = False
                break

        return all_within_eps
        # return (abs(self.constraint(x, y)) <= self.eps).all()

    def _check_optimality_cond(self, x, y, ctx=None):
        if (ctx is None):
            nu = self._get_nu_star(x, y)
        else:
            nu = ctx['nu']
        # nu = self._get_nu_star(x, y) if (ctx is None) else ctx['nu']
        if np.isnan(nu).all():
            return super()._check_optimality_cond(x, y)

        if (abs(self.hY(x, y)) <= self.eps).all():
            warnings.warn("gradient of constraint function vanishes at the optimum.")
            return True

        success = (abs(self.fY(x, y) - np.dot(nu.T, self.hY(x, y))) <= self.eps).all()
        if not success:
            warnings.warn("non-zero Lagrangian gradient {} at y={}, fY={}, hY={}, nu={}".format(
                (self.fY(x, y) - np.dot(nu.T, self.hY(x, y))), y, self.fY(x, y), self.hY(x, y), nu))

        return success



class GeneralConstDeclarativeNode(AbstractDeclarativeNode):

    def __init__(self, dim_x, dim_y):
        super().__init__(dim_x, dim_y)
        self.fY = grad(self.objective, 1)
        self.fYY = jacobian(self.fY, 1)
        self.fXY = jacobian(self.fY, 0)

    def eq_constraints(self, x, y):
        warnings.warn("There are no equality constraints.")
        return None

    def ineq_constraints(self, x, y):
        warnings.warn("There are no inequality constraints.")
        return None

    def gradient(self, x, y=None, ctx=None):
        if y is None:
            y, ctx = self.solve(x)
            # assert self._check_eq_constraints(x, y)
            # assert self._check_ineq_constraints(x, y)
            # assert self._check_optimality_cond(x, y, ctx)

        h_hatY, h_hatX, h_hatYY, h_hatXY = self._get_constraint_derivatives(x, y)

        if (ctx is None or 'nu' not in ctx):
            nu = self._get_nu_star(x, y, h_hatY)
        else:
            nu = ctx['nu']

        # nu = self._get_nu_star(x, y, h_hatY) if (ctx is None or 'nu' not in ctx) else ctx['nu']
        if nu.any() is None or nu.any() == float('-inf'):
            warnings.warn("non-regular solution.")

        # p_plus_q = len(h_hatY)

        H_temp = self.fYY(x, y) - np.sum(nu[i] * h_hatYY[i, :, :] for i in range(len(h_hatY)))  # m-by-m
        H = (H_temp + H_temp.T) / 2   # make sure H is symmetric

        A = h_hatY   # (p+q)-by-m
        B = self.fXY(x, y) - np.sum(nu[i] * h_hatXY[i, :, :] for i in range(len(h_hatY)))  # m-by-n
        C = h_hatX   # (p+q)-by-n


        # try to use cholesky to solve H^{-1}A^T and H^-1 B
        try:
            CC, L = sci.linalg.cho_factor(H)
            invHAT = sci.linalg.cho_solve((CC, L), A.T)
            invHB = sci.linalg.cho_solve((CC, L), B)
        # if H is not positive definite, revert to LU to solve
        except:
            invHAT = sci.linalg.solve(H, A.T)
            invHB = sci.linalg.solve(H, B)

        # compute Dy(x) = H^{-1}A^T(AH^{-1}A^T)^{-1}(AH^{-1}B-C) - H^{-1}B
        # return np.dot(invHAT, sci.linalg.solve(np.dot(A, invHAT), np.dot(A, invHB) - C)) - invHB
        result = np.dot(invHAT, sci.linalg.solve(np.dot(A, invHAT), np.dot(A, invHB) - C)) - invHB
        return result

    def _get_constraint_derivatives(self, x, y):
        h = self.eq_constraints(x, y)   # p-by-1
        if h is not None:
            self._check_eq_constraints(x, y)

        g = self.ineq_constraints(x, y) # q-by-1
        if g is not None:
            self._check_ineq_constraints(x, y)

            # identify active constraints
            mask = np.array([abs(g[i]) <= self.eps for i in range(len(g))])
            if not mask.any():
                mask = None
        else:
            mask = None

        # construct gradient
        if (h is not None) and (mask is None):
            h_hatY = jacobian(self.eq_constraints, 1)(x, y)
            h_hatX = jacobian(self.eq_constraints, 0)(x, y)
            h_hatYY = jacobian(jacobian(self.eq_constraints, 1), 1)(x, y)
            h_hatXY = jacobian(jacobian(self.eq_constraints, 1), 0)(x, y)
            return h_hatY, h_hatX, h_hatYY, h_hatXY

        elif (h is None) and (mask is not None):
            h_hatY = jacobian(self.ineq_constraints, 1)(x, y)[mask]
            h_hatX = jacobian(self.ineq_constraints, 0)(x, y)[mask]
            h_hatYY = jacobian(jacobian(self.ineq_constraints, 1), 1)(x, y)[mask]
            h_hatXY = jacobian(jacobian(self.ineq_constraints, 1), 0)(x, y)[mask]
            return h_hatY, h_hatX, h_hatYY, h_hatXY

        elif (h is not None) and (mask is not None):
            h_hatY = np.vstack((jacobian(self.eq_constraints, 1)(x, y), jacobian(self.ineq_constraints, 1)(x, y)[mask]))
            h_hatX = np.vstack((jacobian(self.eq_constraints, 0)(x, y), jacobian(self.ineq_constraints, 0)(x, y)[mask]))
            h_hatYY = np.vstack((jacobian(jacobian(self.eq_constraints, 1), 1)(x, y), jacobian(jacobian(self.ineq_constraints, 1), 1)(x, y)[mask]))
            h_hatXY = np.vstack((jacobian(jacobian(self.eq_constraints, 1), 0)(x, y), jacobian(jacobian(self.ineq_constraints, 1), 0)(x, y)[mask]))
            return h_hatY, h_hatX, h_hatYY, h_hatXY

        else:
            h_hatY, h_hatX, h_hatYY, h_hatXY = None, None, None, None
            return h_hatY, h_hatX, h_hatYY, h_hatXY


    def _get_nu_star(self, x, y, h_hatY):
        nu = sci.linalg.lstsq(h_hatY.T, self.fY(x, y))[0]
        return nu

    def _check_eq_constraints(self, x, y):
        h = self.eq_constraints(x, y)
        all_within_eps = True
        # Iterate over each element and check the condition
        for i in range(len(h)):
            if abs(h[i]) > self.eps:
                all_within_eps = False
                break
        if (h is None) or all_within_eps:
            return True
        #return (h is None) or (abs(h) <= self.eps).all()
        return False

    def _check_ineq_constraints(self, x, y):
        g = self.ineq_constraints(x, y)
        all_within_eps = True
        # Iterate over each element and check the condition
        for i in range(len(g)):
            if abs(g[i]) > self.eps:
                all_within_eps = False
                break
        if (g is None) or all_within_eps:
            return True
        # if (g is None) or (g <= self.eps).all():
        #     return True
        # return (g is None) or (g <= self.eps).all()
        return False

    def _check_optimality_cond(self, x, y, ctx=None):

        h_hatY = self._get_constraint_derivatives(x, y)[0]
        if h_hatY is None:
            return super()._check_optimality_cond(x, y)

        if (ctx is None):
            nu = self._get_nu_star(x, y, h_hatY)
        else:
            nu = ctx['nu']
        # nu = self._get_nu_star(x, y, h_hatY) if (ctx is None) else ctx['nu']
        if np.isnan(nu).all():
            return super()._check_optimality_cond(x, y)

        all_within_eps = True
        # Iterate over each element and check the condition
        for i in range(len(h_hatY)):
            if abs(h_hatY[i]) > self.eps:
                all_within_eps = False
                break
        # check for invalid lagrangian (gradient of constraint zero at optimal point)
        if all_within_eps:
            warnings.warn("gradient of constraint function vanishes at the optimum.")
            return True

        eq = self.fY(x, y) - np.dot(nu.T, h_hatY)
        all_within_eps = True
        for i in range(len(eq)):
            if abs(eq[i]) > self.eps:
                all_within_eps = False
                break
        # success = (abs(self.fY(x, y) - np.dot(nu.T, h_hatY)) <= self.eps).all()
        if not all_within_eps:
            warnings.warn("non-zero Lagrangian gradient {} at y={}, fY={}, hY={}, nu={}".format(
                (self.fY(x, y) - np.dot(nu.T, h_hatY)), y, self.fY(x, y), h_hatY, nu))

        return all_within_eps