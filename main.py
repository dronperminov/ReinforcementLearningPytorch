import json
import argparse
from visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(description="Reinforcement learning sandbox")
    parser.add_argument("config", type=str, help="Path to configuration file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    visualizer = Visualizer(config)
    visualizer.run()


if __name__ == '__main__':
    main()
