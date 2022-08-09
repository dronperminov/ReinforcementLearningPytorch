from typing import List, Union, Tuple
import torch


class Model(torch.nn.Module):
    def __init__(self, configuration: List[dict], input_shape: Union[int, Tuple[int, int, int]]):
        super(Model, self).__init__()

        layers = []
        self.output_shape = input_shape
        print(self.output_shape)
        for layer_config in configuration:
            layer_type = layer_config["type"]

            if layer_type == "dense":
                self.__add_dense(layers, layer_config)
            elif layer_type == "conv":
                self.__add_conv(layers, layer_config)
            elif layer_type == "maxpool":
                self.__add_maxpool(layers, layer_config)
            elif layer_type == "flatten":
                self.__add_flatten(layers, )
            else:
                raise ValueError(f"Unknown layer type \"{layer_type}\"")
            print(layer_config, self.output_shape)

        self.features = torch.nn.Sequential(*layers)

    def __add_activation(self, layers: List[torch.nn.Module], activation: str):
        if activation == "":
            return

        if activation == "relu":
            layers.append(torch.nn.ReLU())
        elif activation == "leaky-relu":
            layers.append(torch.nn.LeakyReLU())
        elif activation == "sigmoid":
            layers.append(torch.nn.Sigmoid())
        elif activation == "tanh":
            layers.append(torch.nn.Tanh())
        elif activation == "softmax":
            layers.append(torch.nn.Softmax(dim=-1))
        else:
            raise ValueError(f"Unknown activation \"{activation}\"")

    def __add_dense(self, layers: List[torch.nn.Module], config: dict):
        size = config["size"]

        layer = torch.nn.Linear(in_features=self.output_shape, out_features=size)
        torch.nn.init.xavier_normal_(layer.weight)
        layers.append(layer)

        self.__add_activation(layers, config.get("activation", ""))
        self.output_shape = size

    def __add_conv(self, layers: List[torch.nn.Module], config: dict):
        fc = config["fc"]
        fs = config["fs"]
        padding = config.get("padding", 0)
        stride = config.get("stride", 1)

        layer = torch.nn.Conv2d(self.output_shape[0], fc, fs, stride, padding)
        torch.nn.init.xavier_normal_(layer.weight)
        layers.append(layer)

        self.__add_activation(layers, config.get("activation", ""))
        d, h, w = self.output_shape
        self.output_shape = [
            fc,
            (h + 2 * padding - fs) // stride + 1,
            (w + 2 * padding - fs) // stride + 1
        ]

    def __add_maxpool(self, layers: List[torch.nn.Module], config: dict):
        scale = config.get('scale', 2)
        layers.append(torch.nn.MaxPool2d(kernel_size=scale))
        d, h, w = self.output_shape
        self.output_shape = [d, h // scale, w // scale]

    def __add_flatten(self, layers: List[torch.nn.Module]):
        layers.append(torch.nn.Flatten())
        d, h, w = self.output_shape
        self.output_shape = d * h * w

    def forward(self, x):
        features = self.features(x)
        return features


class DuelingModel(Model):
    def __init__(self, configuration: List[dict], input_shape: int, output_shape: int, size: int = 128):
        super(DuelingModel, self).__init__(configuration, input_shape)
        self.value = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.output_shape, out_features=size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=size, out_features=1)
        )
        self.advantage = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.output_shape, out_features=size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=size, out_features=output_shape)
        )

    def forward(self, x):
        features = self.features(x)
        value = self.value(features)
        advantage = self.advantage(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class ActorCriticModel(Model):
    def __init__(self, configuration: List[dict], input_shape: int, output_shape: int):
        super(ActorCriticModel, self).__init__(configuration, input_shape)
        self.actor = torch.nn.Linear(in_features=self.output_shape, out_features=output_shape)
        self.actor_head = torch.nn.Softmax(dim=-1)
        self.critic_head = torch.nn.Linear(in_features=self.output_shape, out_features=1)

    def forward(self, x):
        features = self.features(x)
        action_prob = self.actor_head(self.actor(features))
        state_values = self.critic_head(features)

        return action_prob, state_values
