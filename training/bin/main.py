import argparse
import logging
from pathlib import Path

import yaml

CATTUS_ENGINE_TOP = Path(__file__).parent.parent.parent.resolve() / "engine"


def main():
    log_fmt = "%(asctime)s %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt, datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("--config", type=Path, required=True, help="configuration file")
    parser.add_argument("--run-id", type=str, help="Name of this run, default is current time")
    parser.add_argument("--logfile", type=Path, default=None, help="All logs will be printed to file")
    args = parser.parse_args()

    if args.logfile is not None:
        fileHandler = logging.FileHandler(args.logfile)
        fileHandler.setFormatter(logging.Formatter(log_fmt))
        logging.getLogger().addHandler(fileHandler)

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    from cattus_train import Config, train

    train(Config(**config), run_id=args.run_id)


if __name__ == "__main__":
    main()
