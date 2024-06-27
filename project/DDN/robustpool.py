import torch

class Quadratic():

    is_convex = True

    @staticmethod
    def phi(z, alpha = 1.0):
        return 0.5 * torch.pow(z, 2)

    @staticmethod
    def Dy(z, alpha = 1.0):
        return torch.ones_like(z) / (z.size(-1) * z.size(-2))

class PseudoHuber():

    is_convex = True

    @staticmethod
    def phi(z, alpha = 1.0):
        return alpha * alpha * (torch.sqrt(1.0 + torch.pow(z, 2) / (alpha * alpha)) - 1.0)

    @staticmethod
    def Dy(z, alpha = 1.0):
        w = torch.pow(1.0 + torch.pow(z, 2) / (alpha * alpha), -1.5)
        w_sum = w.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).expand_as(w)
        return torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w), w.div(w_sum))

class Huber():

    is_convex = True

    @staticmethod
    def phi(z, alpha = 1.0):
        z = z.abs()
        return torch.where(z <= alpha, 0.5 * torch.pow(z, 2), alpha * (z - 0.5 * alpha))

    @staticmethod
    def Dy(z, alpha = 1.0):
        w = torch.where(z.abs() <= alpha, torch.ones_like(z), torch.zeros_like(z))
        w_sum = w.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).expand_as(w)
        return torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w), w.div(w_sum))

class Welsch():

    is_convex = True

    @staticmethod
    def phi(z, alpha = 1.0):
        return 1.0 - torch.exp(-torch.pow(z, 2) / (2.0 * alpha * alpha))

    @staticmethod
    def Dy(z, alpha = 1.0):
        z2_on_alpha2 = torch.pow(z, 2) / (alpha * alpha)
        w = (1.0 - z2_on_alpha2) * torch.exp(-0.5 * z2_on_alpha2) / (alpha * alpha)
        w_sum = w.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).expand_as(w)
        Dy_at_x = torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w), w.div(w_sum))
        Dy_at_x = torch.clamp(Dy_at_x, -1.0, 1.0)
        return Dy_at_x

class TruncatedQuadratic():

    is_convex = True

    @staticmethod
    def phi(z, alpha = 1.0):
        assert alpha > 0.0, "alpha must be strictly positive (%f <= 0)" % alpha
        z = z.abs()
        phi_at_z = torch.where(z <= alpha, 0.5 * torch.pow(z, 2), 0.5 * alpha * alpha * torch.ones_like(z))
        return phi_at_z

    @staticmethod
    def Dy(z, alpha = 1.0):
        # Derivative of y(x) for the truncated quadratic penalty function
        w = torch.where(z.abs() <= alpha, torch.ones_like(z), torch.zeros_like(z))
        w_sum = w.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).expand_as(w)
        Dy_at_x = torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w), w.div(w_sum))
        return Dy_at_x

class RobustGlobalPool2dFn(torch.autograd.Function):
    """
    A function to globally pool a 2D response matrix using a robust penalty function
    """
    @staticmethod
    def runOptimisation(x, y, method, alpha_scalar):
        with torch.enable_grad():
            opt = torch.optim.LBFGS([y],
                                    lr=1, # Default: 1
                                    max_iter=100, # Default: 20
                                    max_eval=None, # Default: None
                                    tolerance_grad=1e-05, # Default: 1e-05
                                    tolerance_change=1e-09, # Default: 1e-09
                                    history_size=100, # Default: 100
                                    line_search_fn=None # Default: None, Alternative: "strong_wolfe"
                                    )
            def reevaluate():
                opt.zero_grad()
                # Sum cost function across residuals and batch (all fi are positive)
                f = method.phi(y.unsqueeze(-1).unsqueeze(-1) - x, alpha=alpha_scalar).sum()
                f.backward()
                return f
            opt.step(reevaluate)
        return y

    @staticmethod
    def forward(ctx, x, method, alpha):
        input_size = x.size()
        assert len(input_size) >= 2, "input must at least 2D (%d < 2)" % len(input_size)
        alpha_scalar = alpha.item()
        assert alpha.item() > 0.0, "alpha must be strictly positive (%f <= 0)" % alpha.item()
        x = x.detach()
        x = x.flatten(end_dim=-3) if len(input_size) > 2 else x
        # Handle non-convex functions separately
        if method.is_convex:
            # Use mean as initial guess
            y = x.mean([-2, -1]).clone().requires_grad_()
            y = RobustGlobalPool2dFn.runOptimisation(x, y, method, alpha_scalar)
        else:
            # Use mean and median as initial guesses and choose the best
            # ToDo: multiple random starts
            y_mean = x.mean([-2, -1]).clone().requires_grad_()
            y_mean = RobustGlobalPool2dFn.runOptimisation(x, y_mean, method, alpha_scalar)
            y_median = x.flatten(start_dim=-2).median(dim=-1)[0].clone().requires_grad_()
            y_median = RobustGlobalPool2dFn.runOptimisation(x, y_median, method, alpha_scalar)
            f_mean = method.phi(y_mean.unsqueeze(-1).unsqueeze(-1) - x, alpha=alpha_scalar).sum(-1).sum(-1)
            f_median = method.phi(y_median.unsqueeze(-1).unsqueeze(-1) - x, alpha=alpha_scalar).sum(-1).sum(-1)
            y = torch.where(f_mean <= f_median, y_mean, y_median)
        y = y.detach()
        z = (y.unsqueeze(-1).unsqueeze(-1) - x).clone()
        ctx.method = method
        ctx.input_size = input_size
        ctx.save_for_backward(z, alpha)
        return y.reshape(input_size[:-2]).clone()

    @staticmethod
    def backward(ctx, grad_output):
        z, alpha = ctx.saved_tensors
        input_size = ctx.input_size
        method = ctx.method
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Flatten:
            grad_output = grad_output.detach().flatten(end_dim=-1)
            # Use implicit differentiation to compute derivative:
            grad_input = method.Dy(z, alpha) * grad_output.unsqueeze(-1).unsqueeze(-1)
            # Unflatten:
            grad_input = grad_input.reshape(input_size)
        return grad_input, None, None

class RobustGlobalPool2d(torch.nn.Module):
    def __init__(self, method, alpha=1.0):
        super(RobustGlobalPool2d, self).__init__()
        self.method = method
        self.register_buffer('alpha', torch.tensor([alpha]))

    def forward(self, input):
        return RobustGlobalPool2dFn.apply(input,self.method,self.alpha)

    def extra_repr(self):
        return 'method={}, alpha={}'.format(self.method, self.alpha)

