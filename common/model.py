from typing import List, Union
import torch


class Model(torch.nn.Module):
    def __init__(self, configuration: List[Union[dict, List[dict]]], input_shape: int):
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


class ActorCriticModel(Model):
    def __init__(self, configuration: List[Union[dict, List[dict]]], input_shape: int, output_shape: int):
        super(ActorCriticModel, self).__init__(configuration, input_shape)
        self.actor = torch.nn.Linear(in_features=self.output_shape, out_features=output_shape)
        self.actor_head = torch.nn.Softmax(dim=-1)
        self.critic_head = torch.nn.Linear(in_features=self.output_shape, out_features=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        action_prob = self.actor_head(self.actor(x))
        state_values = self.critic_head(x)

        return action_prob, state_values
