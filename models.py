import numpy as np
import torch
import torchsde
from torch import nn
import diffeq_layers 
import torchdiffeq
import utils


class YNetWithSplit(nn.Module):
    def __init__(self, *blocks):
        # Or create the blocks in side this function given the input size and other hparams
        # Each block has a split module at the end, which does t1, t2 = t.chunk(dim=1, chunks=2).
        pass

    def forward(self, x):
        zs = []
        net = x
        for block in self.blocks:
            z1, z2 = block(net)
            zs.append(z2)
            net = z1
        return zs  # Or cat along non-batch dimension.


def make_y_net(input_size,
               blocks=(2, 2, 2),
               activation="softplus",
               verbose=False,
               explicit_params=True,
               hidden_width=128,
               aug_dim=0,
               mode=0):
    
    """This is the bayesian neural network"""

    _input_size = (input_size[0] + aug_dim,) + input_size[1:]
    layers = []

    for i, num_blocks in enumerate(blocks, 1):
        for j in range(1, num_blocks + 1):
            layers.extend(diffeq_layers.make_ode_k3_block_layers(input_size=_input_size,
                                                                 activation=activation,
                                                                 last_activation=i < len(blocks) or j < num_blocks,
                                                                 hidden_width=hidden_width,
                                                                 mode=mode)) # mode 1 for MNIST; 0 for cifar10

            if verbose:
                if i == 1:
                    print(f"y_net (augmented) input size: {_input_size}")
                layers.append(diffeq_layers.Print(name=f"group: {i}, block: {j}"))

        if i < len(blocks):
            layers.append(diffeq_layers.ConvDownsample(_input_size))
            _input_size = _input_size[0] * 4, _input_size[1] // 2, _input_size[2] // 2

    y_net = diffeq_layers.DiffEqSequential(*layers, explicit_params=explicit_params)

    # return augmented input size b/c y net should have same input / output
    return y_net, _input_size


def make_w_net(in_features, hidden_sizes=(1, 64, 1), activation="softplus", inhomogeneous=True):

    """This is the network that evolves the weights"""

    activation = utils.select_activation(activation)
    all_sizes = (in_features,) + tuple(hidden_sizes) + (in_features,)

    if inhomogeneous:
        layers = []
        for i, (in_size, out_size) in enumerate(zip(all_sizes[:-1], all_sizes[1:]), 1):
            layers.append(diffeq_layers.Linear(in_size, out_size))
            if i + 1 < len(all_sizes):
                layers.append(diffeq_layers.DiffEqWrapper(activation()))
            else:  # Last layer needs zero initialization.
                nn.init.zeros_(layers[-1].weight)
                nn.init.zeros_(layers[-1].bias)
        return diffeq_layers.DiffEqSequential(*layers, explicit_params=False)
    
    else:
        layers = []
        for i, (in_size, out_size) in enumerate(zip(all_sizes[:-1], all_sizes[1:]), 1):
            layers.append(nn.Linear(in_size, out_size))
            if i + 1 < len(all_sizes):
                layers.append(activation())
            else:  # Last layer needs zero initialization.
                nn.init.zeros_(layers[-1].weight)
                nn.init.zeros_(layers[-1].bias)
        return diffeq_layers.DiffEqWrapper(nn.Sequential(*layers))


class BaselineYNet(nn.Module):
    def __init__(self, input_size=(3, 32, 32), num_classes=10, activation="softplus", residual=False, hidden_width=128,
                 aug=0):
        super(BaselineYNet, self).__init__()
        y_net, output_size = make_y_net(
            input_size=input_size, explicit_params=False, activation=activation, hidden_width=hidden_width)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(np.prod(output_size)) + aug, num_classes)
        )
        self.y_net = y_net
        self.residual = residual

    def forward(self, y, *args, **kwargs):
        t = y.new_tensor(0.)
        outs = self.y_net(t, y).flatten(start_dim=1)
        if self.residual:
            outs += y.flatten(start_dim=1)
        return self.projection(outs), torch.tensor(0., device=y.device)


# TODO: add STL
class SDENet(torchsde.SDEStratonovich):
    def __init__(self,
                 input_size=(3, 32, 32),
                 blocks=(2, 2, 2),
                 weight_network_sizes=(1, 64, 1),
                 num_classes=10,
                 activation="softplus",
                 verbose=False,
                 inhomogeneous=True,
                 sigma=0.1,
                 hidden_width=128,
                 aug_dim=0,
                 mode=0):
        super(SDENet, self).__init__(noise_type="diagonal")
        self.input_size = input_size
        self.aug_input_size = (aug_dim + input_size[0], *input_size[1:])
        self.aug_zeros_size = (aug_dim, *input_size[1:])
        self.register_buffer('aug_zeros', torch.zeros(size=(1, *self.aug_zeros_size)))

        # Create network evolving state.
        self.y_net, self.output_size = make_y_net(
            input_size=input_size,
            blocks=blocks,
            activation=activation,
            verbose=verbose,
            hidden_width=hidden_width,
            aug_dim=aug_dim,
            mode=mode,
        )
        # Create network evolving weights.
        initial_params = self.y_net.make_initial_params()  # w0.
        flat_initial_params, unravel_params = utils.ravel_pytree(initial_params)
        self.flat_initial_params = nn.Parameter(flat_initial_params, requires_grad=True)
        self.params_size = flat_initial_params.numel()
        print(f"initial_params ({self.params_size}): {flat_initial_params.shape}")
        self.unravel_params = unravel_params
        self.w_net = make_w_net(
            in_features=self.params_size,
            hidden_sizes=weight_network_sizes,
            activation="tanh",
            inhomogeneous=inhomogeneous
        )

        # Final projection layer.
        self.projection = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(int(np.prod(self.output_size)), num_classes), # option: projection w/o ReLU
            nn.Linear(int(np.prod(self.output_size)), 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
            #nn.Softmax(dim=1), # Add softmax activation
        )

        self.register_buffer('ts', torch.tensor([0., 1.]))
        self.sigma = sigma
        self.nfe = 0

    def f(self, t, y: torch.Tensor):
        input_y = y
        self.nfe += 1
        y, w, _ = y.split(split_size=(y.numel() - self.params_size - 1, self.params_size, 1), dim=1) # params_size: 606408

        fy = self.y_net(t, y.reshape((-1, *self.aug_input_size)), self.unravel_params(w.squeeze(0))).reshape(-1).unsqueeze(0)
        nn = self.w_net(t, w)
        fw = nn - w  # hardcoded OU prior on weights w
        fl = (nn ** 2).sum(dim=1, keepdim=True) / (self.sigma ** 2)

        assert input_y.shape == torch.cat([fy, fw, fl], dim=1).shape, f"Want: {input_y.shape} Got: {torch.cat((fy, fw, fl)).shape}. Check nblocks for dataset divisibility.\n"
        return torch.cat([fy, fw, fl], dim=1)#.squeeze(0)

    def g(self, t, y):
        self.nfe += 1
        gy = torch.zeros(size=(y.numel() - self.params_size - 1,), device=y.device)
        gw = torch.full(size=(self.params_size,), fill_value=self.sigma, device=y.device)
        gl = torch.tensor([0.], device=y.device)
        
        return torch.cat([gy, gw, gl], dim=0).unsqueeze(0)

    def make_initial_params(self):
        return self.y_net.make_initial_params()

    def forward(self, y, adjoint=False, dt=0.02, adaptive=False, adjoint_adaptive=False, method="midpoint", rtol=1e-4, atol=1e-3):
        # Note: This works correctly, as long as we are requesting the nfe after each gradient update.
        #  There are obviously cleaner ways to achieve this.
        self.nfe = 0    
        sdeint = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        
        if self.aug_zeros.numel() > 0:  # Add zero channels.
            aug_zeros = self.aug_zeros.expand(y.shape[0], *self.aug_zeros_size)
            y = torch.cat([y, aug_zeros], dim=1) # 235200
        aug_y = torch.cat((y.reshape(-1), self.flat_initial_params, torch.tensor([0.], device=y.device))) # 841609: (235200, 606408, 1)
        aug_y = aug_y[None]
        
        bm = torchsde.BrownianInterval(
            t0=self.ts[0], t1=self.ts[-1], size=aug_y.shape, dtype=aug_y.dtype, device=aug_y.device,
            cache_size=45 if adjoint else 30  # If not adjoint, don't really need to cache.
        )
        
        if adjoint_adaptive:
            _, aug_y1 = sdeint(self, aug_y, self.ts, bm=bm, method=method, dt=dt, adaptive=adaptive, adjoint_adaptive=adjoint_adaptive, rtol=rtol, atol=atol)
        else:
            _, aug_y1 = sdeint(self, aug_y, self.ts, bm=bm, method=method, dt=dt, adaptive=adaptive, rtol=rtol, atol=atol)
        
        
        y1 = aug_y1[:,:y.numel()].reshape(y.size())
        logits = self.projection(y1)

        logqp = .5 * aug_y1[:, -1]
        
        return logits, logqp

    def zero_grad(self) -> None:
        for p in self.parameters(): p.grad = None


class PartialSDEnet(torchsde.SDEStratonovich):

    def __init__(self,
                input_size=(3, 32, 32),
                blocks=(2, 2, 2),
                weight_network_sizes=(1, 64, 1),
                num_classes=10,
                activation="softplus",
                verbose=False,
                inhomogeneous=True,
                sigma=0.1,
                hidden_width=128,
                aug_dim=0,
                timecut=0.1,
                ode_first=False, 
                mode=0):
        
        # Noise type is diagonal means that the noise is independent across dimensions
        super(PartialSDEnet, self).__init__(noise_type="diagonal")

        self.input_size = input_size
        self.aug_input_size = (aug_dim + input_size[0], *input_size[1:])  # (4, 32, 32) from (3, 32, 32)
        self.aug_zeros_size = (aug_dim, *input_size[1:])                  # (1, 32, 32) from (32, 32)
        self.register_buffer('aug_zeros', torch.zeros(size=(1, *self.aug_zeros_size)))

        # Create network evolving state.
        self.y_net, self.output_size = make_y_net(input_size=input_size,        # output size should be the same as input size with n_features+aug_dim
                                                  blocks=blocks,
                                                  activation=activation,
                                                  verbose=verbose,
                                                  hidden_width=hidden_width,
                                                  aug_dim=aug_dim,
                                                  mode=mode) # mode 1 for MNIST; 0 for cifar10
        
        # Create network evolving weights.
        initial_params = self.y_net.make_initial_params()                            # extracts w0 from the y_net
        flat_initial_params, unravel_params = utils.ravel_pytree(initial_params)     # flattens the w0

        self.flat_initial_params = nn.Parameter(flat_initial_params, requires_grad=True)  # makes parameters (weigths of y_net) trainable
        self.params_size = flat_initial_params.numel()                                    # number of parameters
        self.unravel_params = unravel_params      
        print(f"initial_params ({self.params_size}): {flat_initial_params.shape}")
        
        self.w_net = make_w_net(in_features=self.params_size,
                                hidden_sizes=weight_network_sizes,
                                activation="tanh",
                                inhomogeneous=inhomogeneous)

        # Final projection layer.
        self.projection = nn.Sequential(nn.Flatten(),
                                        # nn.Linear(int(np.prod(self.output_size)), num_classes), # option: projection w/o ReLU
                                        nn.Linear(int(np.prod(self.output_size)), 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024, num_classes),
                                        #nn.Softmax(dim=1),
                                        ) # Add softmax activation

        # Initialise time steps and set forward method
        self.timecut = timecut
        if ode_first:
            self.register_buffer('ts', torch.tensor([0., 1-self.timecut, 1.]))
            self.forward = self.forward_ode_first
        else:
            self.register_buffer('ts', torch.tensor([0., self.timecut, 1.]))
            self.forward = self.forward_sde_first

        # Initialise sigma (noise std)
        self.sigma = sigma
        
        # Initialise number of function evaluations
        self.nfe = 0

    def f(self, t, y: torch.Tensor):
        """
        This is the drift that in this case is common to SDE and ODE 
        """
        input_y = y
        self.nfe += 1

        if self.sde_loop:
            y, w, _ = y.split(split_size=(y.numel() - self.params_size - 1, self.params_size, 1), dim=1)
        else:
            y, w = y.split(split_size=(y.numel() - self.params_size, self.params_size), dim=-1) # change

        # Compute next activation 
        fy = self.y_net(t, y.reshape((-1, *self.aug_input_size)), self.unravel_params(w.squeeze(0))).reshape(-1).unsqueeze(0)
        
        # Compute next weight
        nn = self.w_net(t, w)
        fw = nn - w  # hardcoded OU prior on weights w
        
        if self.sde_loop:
            # Compute next u^2 for divergence control: (prior - posterior) / sigma = (w - (nn-w)) -  this is partial logqp
            fl = ((nn ** 2).sum(dim=1, keepdim=True) / (self.sigma ** 2))
            
            assert input_y.shape == torch.cat([fy, fw, fl], dim=-1).shape, f"Want: {input_y.shape} Got: {torch.cat((fy, fw, fl)).shape}. Check nblocks for dataset divisibility.\n"
            return torch.cat([fy, fw, fl], dim=1)

        else:
            assert input_y.squeeze(0).shape == torch.cat([fy.squeeze(0), fw]).shape
            return torch.cat([fy.squeeze(0), fw]) # change


    def g(self, t, y):
        self.nfe += 1
        gy = torch.zeros(size=(y.numel() - self.params_size - 1,), device=y.device)
        gw = torch.full(size=(self.params_size,), fill_value=self.sigma, device=y.device)
        gl = torch.tensor([0.], device=y.device)
        return torch.cat([gy, gw, gl], dim=0).unsqueeze(0)


    def make_initial_params(self):
        return self.y_net.make_initial_params()


    def forward_sde_first(self, y, adjoint=False, dt=0.02, adaptive=False, adjoint_adaptive=False, method="midpoint", rtol=1e-4, atol=1e-3, return_sde_resuts=False, method_ode="midpoint", rtol_ode=1e-4, atol_ode=1e-3):
        
        # initialise number of function evaluations and boolean sde_loop
        self.nfe = 0  
        self.sde_loop = True

        sdeint = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        
        # Pointless but yeah
        if self.aug_zeros.numel() > 0:  # Add zero channels.
            aug_zeros = self.aug_zeros.expand(y.shape[0], *self.aug_zeros_size)
            y = torch.cat([y, aug_zeros], dim=1) # 235200

        aug_y = torch.cat((y.reshape(-1), self.flat_initial_params, torch.tensor([0.], device=y.device))) # 841609: (235200, 606408, 1)
        aug_y = aug_y[None] # adds a  dimension at the beginning

        # Initialise Brownian motion
        bm = torchsde.BrownianInterval(t0=self.ts[0], t1=self.ts[1], size=aug_y.shape, dtype=aug_y.dtype, device=aug_y.device,
                                       cache_size=45 if adjoint else 30)  # If not adjoint, don't really need to cache.
        
        if adjoint_adaptive:
            _, aug_y1 = sdeint(self, aug_y, self.ts[:2], bm=bm, method=method, dt=dt, adaptive=adaptive, adjoint_adaptive=adjoint_adaptive, rtol=rtol, atol=atol)
        else:
            _, aug_y1 = sdeint(self, aug_y, self.ts[:2], bm=bm, method=method, dt=dt, adaptive=adaptive, rtol=rtol_ode, atol=atol_ode)
        
        # Compute partial logqp
        logqp = .5 * aug_y1[:, -1]

        self.sde_loop = False
        timesteps_ode = torch.linspace(self.ts[1], 1, int((1-self.ts[1])//dt)).to(aug_y.device)

        aug_y2 = odeint(self.f, aug_y[:, :-1].squeeze(0), timesteps_ode, method=method_ode, rtol=rtol_ode, atol=atol_ode)

        # Extract activations after sde integration
        y2 = aug_y2[-1, :y.numel()].reshape(y.size()) # Change

        # Compute logits and logqp
        logits = self.projection(y2)
        
        return logits, logqp
    

    def forward_ode_first(self, y, adjoint=False, dt=0.02, adaptive=False, adjoint_adaptive=False, method="midpoint", rtol=1e-4, atol=1e-3, return_sde_resuts=False, method_ode="midpoint", rtol_ode=1e-4, atol_ode=1e-3):
        
        # initialise number of function evaluations and boolean sde_loop
        self.nfe = 0  
 

        sdeint = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        
        # Pointless but yeah
        if self.aug_zeros.numel() > 0:  # Add zero channels.
            aug_zeros = self.aug_zeros.expand(y.shape[0], *self.aug_zeros_size)
            y = torch.cat([y, aug_zeros], dim=1) # 235200

        aug_y_ode = torch.cat((y.reshape(-1), self.flat_initial_params)) # 841609: (235200, 606408, 1)
        aug_y_ode = aug_y_ode[None] # adds a  dimension at the beginning

        self.sde_loop = False
        timesteps_ode = torch.linspace(self.ts[0], self.ts[1], int(self.ts[1]//dt)).to(aug_y_ode.device)

        aug_y1 = odeint(self.f, aug_y_ode.squeeze(0), timesteps_ode, method=method_ode, rtol=rtol_ode, atol=atol_ode)

        self.sde_loop = True
        aug_y_sde = torch.cat((aug_y1[-1,:].squeeze(0), torch.tensor([0.], device=y.device))).unsqueeze(0)

        # Initialise Brownian motion
        bm = torchsde.BrownianInterval(t0=self.ts[1], t1=self.ts[2], size=aug_y_sde.shape, dtype=aug_y_sde.dtype, device=aug_y_sde.device,
                                       cache_size=45 if adjoint else 30)  # If not adjoint, don't really need to cache.
        
        if adjoint_adaptive:
            _, aug_y2 = sdeint(self, aug_y_sde, self.ts[1:], bm=bm, method=method, dt=dt, adaptive=adaptive, adjoint_adaptive=adjoint_adaptive, rtol=rtol, atol=atol)
        else:
            _, aug_y2 = sdeint(self, aug_y_sde, self.ts[1:], bm=bm, method=method, dt=dt, adaptive=adaptive, rtol=rtol_ode, atol=atol_ode)
        
        y1 = aug_y2[:,:y.numel()].reshape(y.size())
        logits = self.projection(y1)

        logqp = .5 * aug_y2[:, -1] # changed from aug_y1 to aug_y2
        return logits, logqp

    def zero_grad(self) -> None:
        for p in self.parameters(): p.grad = None


if __name__ == "__main__":
    batch_size = 2
    input_size = c, h, w = 3, 32, 32
    sde = SDENet(inhomogeneous=False, input_size=input_size, aug_dim=1)
    sde.ts = torch.tensor([0., 1e-9])  # t0 can't be equal to t1 due to torchsde internal checks, set t1 to be tiny.

    y0 = torch.randn(batch_size, c, h, w)
    y1 = sde(y0)
    torch.testing.assert_close(y0, y1)
