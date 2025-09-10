import argparse
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    trainer = Trainer(args.config)
    trainer.train()

if __name__ == "__main__":
    main()