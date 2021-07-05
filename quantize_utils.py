import torch
def convert_to_nn_module(net):
    out_net = torch.nn.Sequential()
    for name, module in net.named_children():
        if module.__class__.__name__ == 'BatchLinear':
            linear_module = torch.nn.Linear(
                module.in_features,
                module.out_features,
                bias=True if module.bias is not None else False)
            linear_module.weight.data = module.weight.data.clone()
            linear_module.bias.data = module.bias.data.clone()
            out_net.add_module(name, linear_module)
        elif module.__class__.__name__ == 'Sine':
            out_net.add_module(name, module)

        elif module.__class__.__name__ == 'MetaSequential':
            new_module = convert_to_nn_module(module)
            out_net.add_module(name, new_module)
        else:
            if len(list(module.named_children())):
                out_net.add_module(name, convert_to_nn_module(module))
            else: out_net.add_module(name, module)
    return out_net

def convert_to_nn_module_in_place(net):

    for name, module in net.named_children():
        if module.__class__.__name__ == 'BatchLinear':
            linear_module = torch.nn.Linear(
                module.in_features,
                module.out_features,
                bias=True if module.bias is not None else False)
            linear_module.weight.data = module.weight.data.clone()
            linear_module.bias.data = module.bias.data.clone()
            setattr(net, name, linear_module)

        elif module.__class__.__name__ == 'MetaSequential':
            new_module = convert_to_nn_module(module)
            setattr(net, name, new_module)
        else:
            if len(list(module.named_children())):
                new_module = convert_to_nn_module(module)
                setattr(net, name, new_module)

    return net