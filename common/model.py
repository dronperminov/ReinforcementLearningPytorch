from typing import List
import torch


class Model(torch.nn.Module):
    def __init__(self, configuration: List[dict], input_shape: int):
        super(Model, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.output_shape = input_shape

        for layer_config in configuration:
            layer_type = layer_config["type"]

            if layer_type == "dense":
                self.__add_dense(layer_config["size"], layer_config.get("activation", ""))
            else:
                raise ValueError(f"Unknown layer type \"{layer_type}\"")

    def __add_dense(self, size: int, activation: str = ""):
        self.layers.append(torch.nn.Linear(in_features=self.output_shape, out_features=size))

        if activation == "relu":
            self.layers.append(torch.nn.ReLU())
        elif activation == "sigmoid":
            self.layers.append(torch.nn.Sigmoid())
        elif activation == "tanh":
            self.layers.append(torch.nn.Tanh())
        elif activation == "softmax":
            self.layers.append(torch.nn.Softmax(dim=-1))

        self.output_shape = size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
