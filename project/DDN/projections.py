import torch
import torch.nn.functional as F

class Simplex():
    @staticmethod
    def project(v, z = 1.0):
        # 1. Sort v into mu (decreasing)
        mu, _ = v.sort(dim = -1, descending = True)
        # 2. Find rho (number of strictly positive elements of optimal solution w)
        mu_cumulative_sum = mu.cumsum(dim = -1)
        rho = torch.sum(mu * torch.arange(1, v.size()[-1] + 1, dtype=v.dtype, device=v.device) > (mu_cumulative_sum - z), dim = -1, keepdim=True)
        # 3. Compute the Lagrange multiplier theta associated with the simplex constraint
        theta = (torch.gather(mu_cumulative_sum, -1, (rho - 1)) - z) / rho.type(v.dtype)
        # 4. Compute projection
        w = (v - theta).clamp(min = 0.0)
        return w, None

    @staticmethod
    def gradient(grad_output, output, input, is_outside = None):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Flatten:
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        input = input.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        # 2. Use implicit differentiation to compute derivative
        # Select active positivity constraints
        mask = torch.where(output > 0.0, torch.ones_like(input), torch.zeros_like(input))
        masked_output = mask * grad_output
        grad_input = masked_output - mask * (
            masked_output.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True))
        # 3. Unflatten:
        grad_input = grad_input.reshape(output_size)
        return grad_input

class L1Sphere(Simplex):
    @staticmethod
    def project(v, z = 1.0):
        # 1. Take the absolute value of v
        u = v.abs()
        # 2. Project u onto the positive simplex
        beta, _ = Simplex.project(u, z=z)
        # 3. Correct the element signs
        w = beta * torch.where(v < 0, -torch.ones_like(v), torch.ones_like(v))
        return w, None

    @staticmethod
    def gradient(grad_output, output, input, is_outside = None):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Flatten:
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        # 2. Use implicit differentiation to compute derivative
        DYh = output.sign()
        grad_input = DYh.abs() * grad_output - DYh * (
            (DYh * grad_output).sum(-1, keepdim=True) / (DYh * DYh).sum(-1, keepdim=True))
        # 3. Unflatten:
        grad_input = grad_input.reshape(output_size)
        return grad_input

class L2Sphere():
    @staticmethod
    def project(v, z = 1.0):
        # Replace v = 0 with unit vector:
        mask = torch.isclose(v, torch.zeros_like(v), rtol=0.0, atol=1e-12).sum(dim=-1, keepdim=True) == v.size(-1)
        unit_vector = torch.ones_like(v).div(torch.ones_like(v).norm(p=2, dim=-1, keepdim=True))
        v = torch.where(mask, unit_vector, v)
        # Compute projection:
        w = z * v.div(v.norm(p=2, dim=-1, keepdim=True))
        return w, None

    @staticmethod
    def gradient(grad_output, output, input, is_outside = None):
        # 1. Flatten:
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        input = input.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        # 2. Use implicit differentiation to compute derivative
        output_norm = output.norm(p=2, dim=-1, keepdim=True)
        input_norm = input.norm(p=2, dim=-1, keepdim=True)
        ratio = output_norm.div(input_norm)
        grad_input = ratio * (grad_output - output * (
            output * grad_output).sum(-1, keepdim=True).div(output_norm.pow(2)))
        # 3. Unflatten:
        grad_input = grad_input.reshape(output_size)
        return grad_input


class LInfSphere():
    @staticmethod
    def project(v, z = 1.0):
        # 1. Take the absolute value of v
        u = v.abs()
        # 2. Project u onto the (non-negative) LInf-sphere
        # If u_i >= z, u_i = z
        # If u_i < z forall i, find max and set to z
        z = torch.tensor(z, dtype=v.dtype, device=v.device)
        u = torch.where(u.gt(z), z, u)
        u = torch.where(u.ge(u.max(dim=-1, keepdim=True)[0]), z, u)
        # 3. Correct the element signs
        w = u * torch.where(v < 0, -torch.ones_like(v), torch.ones_like(v))
        return w, None

    @staticmethod
    def gradient(grad_output, output, input, is_outside = None):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Flatten:
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        # 2. Use implicit differentiation to compute derivative
        mask = output.abs().ge(output.abs().max(dim=-1, keepdim=True)[0])
        hY = output.sign() * mask.type(output.dtype)
        grad_input = grad_output - hY.abs() * grad_output
        # 3. Unflatten:
        grad_input = grad_input.reshape(output_size)
        return grad_input


class EuclideanProjectionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, method, radius):
        output, is_outside = method.project(input, radius.item())
        ctx.method = method
        ctx.save_for_backward(output.clone(), input.clone(), is_outside)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, input, is_outside = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = ctx.method.gradient(grad_output, output, input, is_outside)
        return grad_input, None, None


class EuclideanProjection(torch.nn.Module):
    def __init__(self, method, radius = 1.0):
        super(EuclideanProjection, self).__init__()
        self.method = method
        self.register_buffer('radius', torch.tensor([radius]))

    def forward(self, input):
        return EuclideanProjectionFn.apply(input, self.method, self.radius)

    def extra_repr(self):
        return 'method={}, radius={}'.format(self.method.__name__, self.radius)
