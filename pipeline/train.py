from yaml import safe_load
import argparse
from model.model import CustomStableDiffusionTraining

def train(configs):
    """
    This is the main train loop, which will be called by accelerate
    for distributed training
    """
    model = CustomStableDiffusionTraining(configs)
    model.train()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--configs", type=str,
                               default="../configs/experiment_1.yaml",
                                help=(
                                    "The path the configs.yaml for the model training"
                                ),)
    args = parser.parse_args()

    with open(args.configs) as infile:
        configs = safe_load(infile)

    train(configs)