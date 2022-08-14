# ReinforcementLearningPytorch
Реализация различных алгоритмов обучения с подкреплением с использованием фреймворка Pytorch

## Реализованные алгоритмы

### Алгоритмы Q-learning
* [Deep Q network (DQN)](https://github.com/dronperminov/ReinforcementLearningPytorch/blob/master/algorithms/dqn.py)
* [Double deep Q network (DDQN)](https://github.com/dronperminov/ReinforcementLearningPytorch/blob/master/algorithms/dqn.py)
* [Dueling deep Q network (Dueling DQN)](https://github.com/dronperminov/ReinforcementLearningPytorch/blob/master/algorithms/dqn.py)
* [Noisy deep Q network (Noisy DQN)](https://github.com/dronperminov/ReinforcementLearningPytorch/blob/master/algorithms/dqn.py)
* в том числе soft версия и использование приоритизированного буфера ([PER](https://github.com/dronperminov/ReinforcementLearningPytorch/blob/master/common/prioritized_replay_buffer.py))

### Алгоритмы оптимизации политики (policy optimization)
* [Reinforce](https://github.com/dronperminov/ReinforcementLearningPytorch/blob/master/algorithms/reinforce.py)
* [Actor critic (AC)](https://github.com/dronperminov/ReinforcementLearningPytorch/blob/master/algorithms/actor_critic.py)
* [Advantage actor critic (A2C)](https://github.com/dronperminov/ReinforcementLearningPytorch/blob/master/algorithms/a2c.py)
* [Generalized advantage estimation (GAE)](https://github.com/dronperminov/ReinforcementLearningPytorch/blob/master/algorithms/gae.py)
* [Proximal policy optimization (PPO)](https://github.com/dronperminov/ReinforcementLearningPytorch/blob/master/algorithms/ppo.py)


## Используемые среды

В настоящий момент доступна только одна среда - игра [змейка](https://github.com/dronperminov/ReinforcementLearningPytorch/blob/master/environments/snake.py). Доступна настройка размеров поля и режима состояния игры - в виде вектора или же в виде последовательного набора кадров самого поля.
Пример настройки среды для игры на поле 10х15 с состоянием в виде 4 кадров:
```json
"environment": {
    "name": "snake",
    "config": {
        "field_width": 15,
        "field_height": 10,
        "frames_count": 4
    }
}
```

## Конфигурация игры

Для запуска экспериментов необходимо подготовить конфигурационный json файл следующего вида:
```json
{
    "environment": {
        "параметры_среды"
    },

    "view_config": {
        "параметры отображения"
    },

    "algorithms": [
        {
            "параметры алгоритма 1"
        },

        {
            "параметры алгоритма 2"
        },

        {
            "параметры алгоритма 3"
        }
    ]
}
```

Примеры конфигурационных файлов для различных алгоритмов доступны в папке [configs](https://github.com/dronperminov/ReinforcementLearningPytorch/blob/master/configs).

### Параметры среды

Среда задаётся в виде словаря, содержащего имя среды `name` и её параметры в виде конфигурационного словаря `config`:

```json
"environment": {
    "name": "snake",
    "config": {
        "параметры"
    }
}
```

### Параметры отображения

Словарь, состоящий из следующих параметров:
* `use_gui` - флаг использования графического окна
* `width` - ширина области просмотра среды
* `height` - высота области просмотра среды
* `plot_rewards` - отрисовывать ли график наград
* `plot_keys` - список ключей для построения графика по информации среды/алгоритма
* `info_keys` - список ключей для отображения из информации среды/алгоритма
* `columns` - количество столбцов для задания сетки (при использовании более одного алгоритма)
* `caption` - заголовок графического окна

#### Пример параметров отображения

```json
"view_config": {
    "use_gui": true,
    "width": 302,
    "height": 202,
    "plot_rewards": true,
    "plot_keys": ["length"],
    "info_keys": ["wall", "eat_self", "no_eat", "length", "max_length"],
    "columns": 1,
    "caption": "Snake (convolution network)"
}
```

### Параметры алгоритмов

Сами алгоритмы (ключ `algorithms`) - это список словарей, определяющих параметры запускаемых алгоритмов. Для запуска всего одного алгоритма список будет состоять лишь из одного словаря. В параметрах алгоритма обязательно должен быть ключ `name`, определяющий, какой алгоритм будет запускаться. При желании можно также задать дополнительные параметры алгоритма.

В случае запуска множества алгоритмов, использующих общие параметры (например, конфигурация нейронной сети, оптимизатор, seed и т.д.) можно в конфигурационном файле задать словарь с ключом `common_algorithms_params`, в котором , аналогично словарю с параметрами алгоритма, перечислить необходимые параметры:

```json
"common_algorithms_params": {
    "name": "ppo",
    "gamma": 0.99,
    "seed": 42,

    "optimizer": "adam",
    "learning_rate": 0.001,

    "agent_architecture": [
        {"type": "dense", "size": 256, "activation": "leaky-relu"},
        {"type": "dense", "size": 256, "activation": "leaky-relu"}
    ],
},

"algorithms": [
    {
        "ppo_steps": 5,
        "ppo_clip": 0.2,
    },

    {
        "ppo_steps": 10,
        "ppo_clip": 0.2,
    },

    {
        "ppo_steps": 5,
        "ppo_clip": 0.1,
    }
]
```

## Доступные параметры различных алгоритмов

Для всех алгоритмов общими являются:
* `name` - название алгоритма
* `gamma` - параметр дисконтирования
* `optimizer` - используемый оптимизатор
* `learning_rate` - скорость обучения
* `agent_architecture` - список словарей, задающий конфигурацию сети (до выходного слоя)

Пример конфигурации полносвязной сети:

```json
"agent_architecture": [
    {"type": "dense", "size": 256, "activation": "leaky-relu"},
    {"type": "dense", "size": 256, "activation": "leaky-relu"}
]
```

Пример конфигурации свёрточной сети:

```json
"agent_architecture": [
    {"type": "conv", "fc": 32, "fs": 7, "padding": 3, "activation": "leaky-relu", "stride": 2},
    {"type": "conv", "fc": 64, "fs": 5, "padding": 2, "activation": "leaky-relu", "stride": 2},
    {"type": "conv", "fc": 128, "fs": 3, "padding": 1, "activation": "leaky-relu", "stride": 2},
    {"type": "conv", "fc": 256, "fs": 3, "padding": 1, "activation": "leaky-relu", "stride": 2},
    {"type": "flatten"},
    {"type": "dense", "size": 256, "activation": "leaky-relu"}
]
```

### Deep q network (dqn)

* `ddqn` - использование модификации double deep Q network (`false` по умолчанию)
* `model_type` - тип используемой модели: `dueling` / `noisy` (по умолчанию используется обычная `dqn`)
* `head_size` - размер полносвязного слоя, находящегося перед выходным (по умолчанию `0`)
* `use_per` - использование приоритизированного буфера опыта (`false` по умолчанию)
* `batch_size` - размер батча для обучения сети (`128` по умолчанию)
* `min_replay_size` - минимальный размер буфера опыта до начала обучения (`1000` по умолчанию)
* `max_replay_size` - максимальный размер буфера опыта (`10000` по умолчанию)
* `max_epsilon` - начальное значение epslilon для epsilon-greedy алгоритма (`1` по умолчанию)
* `min_epsilon` - конечное значение epslilon для epsilon-greedy алгоритма (`0.01` по умолчанию)
* `decay` - шаг изменения epsilon по экспоненциальному закону (`0.004` по умолчанию)
* `train_model_period` - как часто выполнять обучение модели (`4` по умолчанию)
* `update_target_model_period` - как часто обновлять целевую модель (`500` по умолчанию)
* `tau` - если задан, то определяет коэффициент мягкого (soft) обновления параметров целевой модели (`1e-3` рекомендуется)


### Generalized advantage estimation (gae)

* `trace_decay` - параметр дисконтирования выгоды (`0.99` по умолчанию)


### Proximal policy optimization (ppo)

* `ppo_steps` - количество шагов обучения модели (`5` по умолчанию)
* `ppo_clip` - коэффициент отсечения PPO loss (`0.2` по умолчанию)