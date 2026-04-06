from selectq.config import load_config
from selectq.pipeline import run_pipeline


def main() -> None:
    cfg = load_config()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
