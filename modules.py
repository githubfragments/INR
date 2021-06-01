import torch
from torch import nn
import sys
sys.path.append('siren')
from siren.modules import Sine, RBFLayer,  ImageDownsampling, FCBlock, BatchLinear
from torchmeta.modules import MetaModule, MetaSequential
import numpy as np
from collections import OrderedDict
from torchmeta.modules.utils import get_subdict
from siren.modules import sine_init, first_layer_sine_init, init_weights_normal, init_weights_xavier, init_weights_elu, init_weights_selu
import math
import torch.nn.functional as F

class FCBlock_BN(MetaModule):
    '''Adds Batch Norm to SIRENS FCBlock.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), torch.nn.BatchNorm1d(hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), torch.nn.BatchNorm1d(hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())
        coords = coords.squeeze()
        output = self.net(coords, params=get_subdict(params, 'net'))
        output = output.unsqueeze(0)
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations


class SingleBVPNet_INR(MetaModule):
    '''A canonical representation network for a BVP with additional support for Fourier Featrues'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, batch_norm=False, ff_dims=None, **kwargs):
        super().__init__()
        self.mode = mode
        num_frequencies = None
        if ff_dims and len(ff_dims) == 1:
            num_frequencies = int(ff_dims[0])
        if self.mode == 'rbf':
            self.rbf_layer = RBFLayer(in_features=in_features, out_features=kwargs.get('rbf_centers', 1024))
            in_features = kwargs.get('rbf_centers', 1024)
        elif self.mode == 'nerf':
            self.positional_encoding = PosEncodingNeRF(in_features=in_features,
                                                       sidelength=kwargs.get('sidelength', None),
                                                       fn_samples=kwargs.get('fn_samples', None),
                                                       use_nyquist=kwargs.get('use_nyquist', True),
                                                       num_frequencies=num_frequencies)
            in_features = self.positional_encoding.out_dim

        elif self.mode == 'positional':
            num_frq = num_frequencies if num_frequencies else hidden_features//2
            self.positional_encoding = FourierFeatureEncodingPositional(in_features=in_features, num_frequencies=num_frq, scale=kwargs.get('encoding_scale',6.0))
            in_features = self.positional_encoding.out_dim
        elif self.mode == 'gauss':
            num_frq = num_frequencies if num_frequencies else hidden_features
            self.positional_encoding = FourierFeatureEncodingGaussian(in_features=in_features, num_frequencies=num_frq, scale=kwargs.get('encoding_scale',6.0))
            in_features = self.positional_encoding.out_dim

        self.image_downsampling = ImageDownsampling(sidelength=kwargs.get('sidelength', None),
                                                    downsample=kwargs.get('downsample', False))
        self.batch_norm = batch_norm
        if batch_norm:
            self.net = FCBlock_BN(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                               hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        else:
            self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)

        #print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        # various input processing methods for different applications
        if self.image_downsampling.downsample:
            coords = self.image_downsampling(coords)
        if self.mode == 'rbf':
            coords = self.rbf_layer(coords)
        elif self.mode == 'nerf':
            coords = self.positional_encoding(coords)
        elif self.mode == 'positional':
            coords = self.positional_encoding(coords)
        elif self.mode == 'gauss':
            coords = self.positional_encoding(coords)

        output = self.net(coords, get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

    def predict(self, model_input):
        return self.forward(model_input)


    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}

class Parallel_INR(nn.Module):
    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=[64, 128, 256], num_hidden_layers=3, batch_norm=False, **kwargs):
        super().__init__()
        self.nets = nn.ModuleList()
        hidden_features.sort()  # sort in ascending order
        self.hidden_features = hidden_features
        for hidden_dim in hidden_features:
            model = SingleBVPNet_INR(out_features=out_features, type=type, in_features=in_features,
                 mode=mode, hidden_features=hidden_dim, num_hidden_layers=num_hidden_layers, batch_norm=batch_norm, **kwargs)
            self.nets.append(model)
        print(self)

    def forward(self, model_input):
        outputs = []
        for net in self.nets:
            outputs.append(net.forward(model_input))
        return outputs

    def predict(self, model_input):
        model_output = self.forward(model_input)
        model_output = {'model_in': model_output[0]['model_in'],
                        'model_out': sum([p['model_out'] for p in model_output])}
        return model_output


class SingleBVPNet_INR_nodetach(MetaModule):
    '''A canonical representation network for a BVP with additional support for Fourier Featrues'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, batch_norm=False, ff_dims=None, **kwargs):
        super().__init__()
        self.mode = mode
        num_frequencies = None
        if ff_dims and len(ff_dims) == 1:
            num_frequencies = int(ff_dims[0])
        if self.mode == 'rbf':
            self.rbf_layer = RBFLayer(in_features=in_features, out_features=kwargs.get('rbf_centers', 1024))
            in_features = kwargs.get('rbf_centers', 1024)
        elif self.mode == 'nerf':
            self.positional_encoding = PosEncodingNeRF(in_features=in_features,
                                                       sidelength=kwargs.get('sidelength', None),
                                                       fn_samples=kwargs.get('fn_samples', None),
                                                       use_nyquist=kwargs.get('use_nyquist', True),
                                                       num_frequencies=num_frequencies)
            in_features = self.positional_encoding.out_dim

        elif self.mode == 'positional':
            num_frq = num_frequencies if num_frequencies else hidden_features//2
            self.positional_encoding = FourierFeatureEncodingPositional(in_features=in_features, num_frequencies=num_frq, scale=kwargs.get('encoding_scale',6.0))
            in_features = self.positional_encoding.out_dim
        elif self.mode == 'gauss':
            num_frq = num_frequencies if num_frequencies else hidden_features
            self.positional_encoding = FourierFeatureEncodingGaussian(in_features=in_features, num_frequencies=num_frq, scale=kwargs.get('encoding_scale',6.0))
            in_features = self.positional_encoding.out_dim

        self.image_downsampling = ImageDownsampling(sidelength=kwargs.get('sidelength', None),
                                                    downsample=kwargs.get('downsample', False))
        self.batch_norm = batch_norm
        if batch_norm:
            self.net = FCBlock_BN(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                               hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        else:
            self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)

        #print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords']#.clone().detach().requires_grad_(True)
        coords = coords_org

        # various input processing methods for different applications
        if self.image_downsampling.downsample:
            coords = self.image_downsampling(coords)
        if self.mode == 'rbf':
            coords = self.rbf_layer(coords)
        elif self.mode == 'nerf':
            coords = self.positional_encoding(coords)
        elif self.mode == 'positional':
            coords = self.positional_encoding(coords)
        elif self.mode == 'gauss':
            coords = self.positional_encoding(coords)

        output = self.net(coords, get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

    def predict(self, model_input):
        return self.forward(model_input)


    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}


class INR_Mixture(nn.Module):
    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, tapered=False, ff_dims=None,num_components=4, **kwargs):
        super().__init__()
        self.mode = mode
        self.use_meta = True
        self.in_features = in_features
        self.num_components = num_components
        self.out_features = out_features
        self.nets = nn.ModuleList()
        self.hidden_features = hidden_features
        self.index = None
        for i in range(num_components):
            model = SingleBVPNet_INR_nodetach(out_features=out_features, type=type, in_features=in_features,
                                     mode=mode, hidden_features=hidden_features, num_hidden_layers=num_hidden_layers, **kwargs)
            self.nets.append(model)


    def forward(self, model_input, type=torch.float16):
        if self.index == None:
            self.index = self.mapping_function(model_input)
        index = self.index


        if self.use_meta:
            # Enables us to compute gradients w.r.t. coordinates
            out = torch.zeros((1, model_input['coords'].shape[1], self.out_features), dtype=type,
                              requires_grad=True).cuda()
            coords_org = model_input['coords'].clone().detach().requires_grad_(True)
            coords = coords_org
        else:
            out = torch.zeros((1, model_input.shape[1], self.out_features), dtype=torch.float32,
                              requires_grad=False).cuda()
            coords_org = model_input
            coords = coords_org

        #inp = torch.zeros_like(coords)
        for comp_idx in range(self.num_components):
            bool_tensor = index == comp_idx
            if self.use_meta:
                sliced_input = {'coords': coords[bool_tensor]}
                out_dict = self.nets[comp_idx](sliced_input)
                out[bool_tensor] = out_dict['model_out'].squeeze()
                ret = {'model_in': coords_org, 'model_out': out}
            else:
                self.index = self.mapping_function({'coords': model_input})
                bool_tensor = self.index == comp_idx
                bool_tensor_i = bool_tensor
                if len(coords.shape) == 2: bool_tensor_i = bool_tensor.squeeze()
                sliced_input = coords[bool_tensor_i]
                out_dict = self.nets[comp_idx](sliced_input)
                out[bool_tensor] = out_dict.squeeze()
                ret = out
            #inp[bool_tensor] = out_dict['model_in']

        return ret



    def mapping_function(self, input):
        num_per_axis = int(torch.pow(torch.tensor(self.num_components), torch.tensor(1 / self.in_features)))
        intervals = torch.linspace(-1, 1, steps=int(torch.pow(torch.tensor(self.num_components), torch.tensor(1/self.in_features))) + 1)
        index = torch.zeros_like(input['coords'][:, :, 0])
        for j in range(self.in_features):
            temp_index = torch.zeros_like(input['coords'][:, :, 0])
            for i in intervals[:-1]: #skip last interval boundary
                temp_index += input['coords'][:,:,j] >= i
            index += (temp_index - 1)* (j * num_per_axis if j > 0 else 1)
        return index
    def predict(self, model_input):

        return self.forward(model_input, type=torch.float32)


class MultiScale_INR(MetaModule):
    '''A canonical representation network for a BVP with additional support for Fourier Featrues'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, tapered=False, ff_dims=None, **kwargs):
        super().__init__()
        self.mode = mode
        self.use_meta = True
        self.ff_dims = ff_dims
        if ff_dims == None:
                if self.mode == 'rbf':
                    self.rbf_layer = RBFLayer(in_features=in_features, out_features=kwargs.get('rbf_centers', 1024))
                    in_features = kwargs.get('rbf_centers', 1024)
                elif self.mode == 'nerf':
                    self.positional_encoding = PosEncodingNeRF(in_features=in_features,
                                                               sidelength=kwargs.get('sidelength', None),
                                                               fn_samples=kwargs.get('fn_samples', None),
                                                               use_nyquist=kwargs.get('use_nyquist', True))
                    in_features = self.positional_encoding.out_dim

                elif self.mode == 'positional':
                    self.positional_encoding = FourierFeatureEncodingPositional(in_features=in_features, num_frequencies=hidden_features//2, scale=kwargs.get('encoding_scale',6.0))
                    in_features = self.positional_encoding.out_dim
                elif self.mode == 'gauss':
                    self.positional_encoding = FourierFeatureEncodingGaussian(in_features=in_features, num_frequencies=hidden_features, scale=kwargs.get('encoding_scale',6.0))
                    in_features = self.positional_encoding.out_dim

                sidelength = kwargs.get('sidelength', None)
                sidelenght2x = (sidelength[0] // 2, sidelength[1] // 2)
                sidelenght4x = (sidelength[0] // 4, sidelength[1] // 4)
                self.image_downsampling2x = ImageDownsampling(sidelength=sidelenght2x,
                                                              downsample=kwargs.get('downsample', False))
                self.image_downsampling4x = ImageDownsampling(sidelength=sidelenght4x,
                                                              downsample=kwargs.get('downsample', False))

                hidden_features1x = hidden_features
                hidden_features2x = hidden_features // 2 if tapered else hidden_features
                hidden_features4x = hidden_features // 4 if tapered else hidden_features
                self.net1x = FCBlock(in_features=in_features + hidden_features2x, out_features=out_features,
                                     num_hidden_layers=num_hidden_layers,
                                     hidden_features=hidden_features1x, outermost_linear=True, nonlinearity=type)
                self.net2x = FCBlock(in_features=in_features + hidden_features4x, out_features=hidden_features2x,
                                     num_hidden_layers=num_hidden_layers,
                                     hidden_features=hidden_features2x, outermost_linear=True, nonlinearity=type)
                self.net4x = FCBlock(in_features=in_features, out_features=hidden_features4x,
                                     num_hidden_layers=num_hidden_layers,
                                     hidden_features=hidden_features4x, outermost_linear=True, nonlinearity=type)




        else:
            ff_dims = [int(s) for s in ff_dims]
            if self.mode == 'nerf':
                self.positional_encoding1x = PosEncodingNeRF(in_features=in_features,
                                                        num_frequencies=ff_dims[0])

                self.positional_encoding2x = PosEncodingNeRF(in_features=in_features,
                                                            num_frequencies=ff_dims[1])

                self.positional_encoding4x = PosEncodingNeRF(in_features=in_features,
                                                            num_frequencies=ff_dims[2])



            elif self.mode == 'positional':
                self.positional_encoding1x = FourierFeatureEncodingPositional(in_features=in_features,
                                                                            num_frequencies=ff_dims[0],
                                                                            scale=kwargs.get('encoding_scale', 6.0))
                self.positional_encoding2x = FourierFeatureEncodingPositional(in_features=in_features,
                                                                              num_frequencies=ff_dims[1],
                                                                              scale=kwargs.get('encoding_scale', 6.0))
                self.positional_encoding4x = FourierFeatureEncodingPositional(in_features=in_features,
                                                                              num_frequencies=ff_dims[2],
                                                                              scale=kwargs.get('encoding_scale', 6.0))


            elif self.mode == 'gauss':
                self.positional_encoding1x = FourierFeatureEncodingGaussian(in_features=in_features,
                                                                              num_frequencies=ff_dims[0],
                                                                              scale=kwargs.get('encoding_scale', 6.0))
                self.positional_encoding2x = FourierFeatureEncodingGaussian(in_features=in_features,
                                                                              num_frequencies=ff_dims[1],
                                                                              scale=kwargs.get('encoding_scale', 6.0))
                self.positional_encoding4x = FourierFeatureEncodingGaussian(in_features=in_features,
                                                                              num_frequencies=ff_dims[2],
                                                                              scale=kwargs.get('encoding_scale', 6.0))
            in_features1x = self.positional_encoding1x.out_dim
            in_features2x = self.positional_encoding2x.out_dim
            in_features4x = self.positional_encoding4x.out_dim
            sidelength = kwargs.get('sidelength', None)
            sidelenght2x = (sidelength[0] // 2, sidelength[1] // 2)
            sidelenght4x = (sidelength[0] // 4, sidelength[1] // 4)
            self.image_downsampling2x = ImageDownsampling(sidelength=sidelenght2x,
                                                          downsample=kwargs.get('downsample', False))
            self.image_downsampling4x = ImageDownsampling(sidelength=sidelenght4x,
                                                          downsample=kwargs.get('downsample', False))

            hidden_features1x = hidden_features
            hidden_features2x = hidden_features // 2 if tapered else hidden_features
            hidden_features4x = hidden_features // 4 if tapered else hidden_features
            self.net1x = FCBlock(in_features=in_features1x + hidden_features2x, out_features=out_features,
                                 num_hidden_layers=num_hidden_layers,
                                 hidden_features=hidden_features1x, outermost_linear=True, nonlinearity=type)
            self.net2x = FCBlock(in_features=in_features2x + hidden_features4x, out_features=hidden_features2x,
                                 num_hidden_layers=num_hidden_layers,
                                 hidden_features=hidden_features2x, outermost_linear=True, nonlinearity=type)
            self.net4x = FCBlock(in_features=in_features4x, out_features=hidden_features4x,
                                 num_hidden_layers=num_hidden_layers,
                                 hidden_features=hidden_features4x, outermost_linear=True, nonlinearity=type)



        #print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        if self.use_meta:
            # Enables us to compute gradients w.r.t. coordinates
            coords_org = model_input['coords'].clone().detach().requires_grad_(True)
            coords = coords_org
        else:
            coords_org = model_input
            coords = coords_org

        # various input processing methods for different applications
        encoding = torch.nn.Identity()
        if self.mode == 'rbf':
            encoding1x = self.rbf_layer
            encoding2x = self.rbf_layer
            encoding4x = self.rbf_layer
        elif self.mode in ['nerf', 'positional', 'gauss']:
            if self.ff_dims is None:
                encoding1x = self.positional_encoding
                encoding2x = self.positional_encoding
                encoding4x = self.positional_encoding
            else:
                encoding1x = self.positional_encoding1x
                encoding2x = self.positional_encoding2x
                encoding4x = self.positional_encoding4x
        input4x = encoding4x(self.image_downsampling4x(coords))
        if self.use_meta:
            output4x = self.net4x(input4x, get_subdict(params, 'net4x'))
        else:
            output4x = self.net4x(input4x)

        input2x = torch.cat([encoding2x(self.image_downsampling2x(coords)), output4x], axis=-1)

        if self.use_meta:
            output2x = self.net2x(input2x, get_subdict(params, 'net2x'))
        else:
            output2x = self.net2x(input2x)


        input1x = torch.cat([encoding1x(coords), output2x], axis=-1)
        if self.use_meta:
            output = self.net1x(input1x, get_subdict(params, 'net1x'))
            res = {'model_in': coords_org, 'model_out': output}
        else:
            output = self.net1x(input1x)
            res = output

        return res

    def predict(self, model_input):
        return self.forward(model_input)

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}


class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True, num_frequencies=None):
        super().__init__()

        self.in_features = in_features
        if num_frequencies == None:
            if self.in_features == 3:
                self.num_frequencies = 10
            elif self.in_features == 2:
                assert sidelength is not None
                if isinstance(sidelength, int):
                    sidelength = (sidelength, sidelength)
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
            elif self.in_features == 1:
                assert fn_samples is not None
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = num_frequencies
        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

class FourierFeatureEncodingPositional(nn.Module):
    '''Module to add fourier features as in Tancik[2020].'''

    def __init__(self, in_features, num_frequencies, scale):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.frequencies = scale**(torch.range(0, num_frequencies-1) / num_frequencies)
        self.frequencies = self.frequencies.cuda()
        self.scale = scale
        self.out_dim = 2 * in_features * self.num_frequencies


    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)
        #
        # coords_pos_enc = torch.tensor([]).cuda()
        # for i in range(self.num_frequencies):
        #     for j in range(self.in_features):
        #         c = coords[..., j]
        #
        #         sin = torch.unsqueeze(torch.sin((self.scale ** (i / self.num_frequencies)) * np.pi * c), -1)
        #         cos = torch.unsqueeze(torch.cos((self.scale ** (i / self.num_frequencies)) * np.pi * c), -1)
        #
        #         coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        coord_freq = torch.einsum('p, sqr->sqrp', self.frequencies, coords)
        sin = torch.sin(2 * np.pi * coord_freq)
        cos = torch.cos(2 * np.pi * coord_freq)
        coords_pos_enc = torch.cat((sin, cos), axis=-1)
        res = coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)
        return res
class FourierFeatureEncodingGaussian(nn.Module):
    '''Module to add Gaussian Fourier features as in Tancik[2020].'''

    def __init__(self, in_features, num_frequencies, scale):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.scale = scale
        self.out_dim = 2 * self.num_frequencies
        self.B = torch.nn.parameter.Parameter(self.scale * torch.randn(self.num_frequencies, self.in_features), requires_grad=False)


    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = torch.tensor([]).cuda()

        cos = torch.cos(2 * np.pi * torch.matmul(coords,torch.transpose(self.B, 0, 1)))
        sin = torch.sin(2 * np.pi * torch.matmul(coords,torch.transpose(self.B, 0, 1)))
        coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc
        # for i in range(self.num_frequencies):
        #     for j in range(self.in_features):
        #         c = coords[..., j]
        #
        #         sin = torch.unsqueeze(torch.sin((self.scale ** (i / self.num_frequencies)) * np.pi * c), -1)
        #         cos = torch.unsqueeze(torch.cos((self.scale ** (i / self.num_frequencies)) * np.pi * c), -1)
        #
        #         coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        #
        # return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)
