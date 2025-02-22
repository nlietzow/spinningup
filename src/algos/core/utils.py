import torch.nn as nn


def mlp(
    sizes: tuple[int, ...],
    activation: type[nn.Module],
    output_activation: type[nn.Module],
) -> nn.Sequential:
    def build():
        for j in range(len(sizes) - 2):
            yield nn.Linear(sizes[j], sizes[j + 1])
            yield activation()

        yield nn.Linear(sizes[-2], sizes[-1])
        yield output_activation()

    return nn.Sequential(*build())


def mlp_bn(
    sizes: tuple[int, ...],
    batch_norm_eps: float,
    batch_norm_momentum: float,
    activation: type[nn.Module],
    output_activation: type[nn.Module],
) -> nn.Sequential:
    def build():
        for j in range(len(sizes) - 2):
            yield nn.Linear(sizes[j], sizes[j + 1])
            yield nn.BatchNorm1d(
                sizes[j + 1],
                eps=batch_norm_eps,
                momentum=batch_norm_momentum,
            )
            yield activation()

        yield nn.Linear(sizes[-2], sizes[-1])
        yield output_activation()

    return nn.Sequential(*build())
