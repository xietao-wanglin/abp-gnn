import argparse
import wandb
from omegaconf import OmegaConf
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.config)
    wandb.login()
    wandb.init(
        project="ABP_GNN",
        name=base_cfg.wandb.name,
        config=OmegaConf.to_container(base_cfg, resolve=True),
    )

    wandb_cfg = wandb.config.as_dict()
    merged_cfg = OmegaConf.merge(base_cfg, OmegaConf.create(wandb_cfg))

    trainer = Trainer(config=merged_cfg, config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
