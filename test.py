"""Evaluate a checkpoint with the recipe's validation or test datasets."""

from pathlib import Path


def test_pipeline(root_path: str) -> None:
    from rich.traceback import install
    from traiNNer.tasks.dispatch import evaluate_holdout
    from traiNNer.utils.config import Config

    install()
    opt, _ = Config.load_config_from_file(root_path, is_train=False)
    try:
        evaluate_holdout(opt)
    except KeyboardInterrupt:
        from traiNNer.utils import get_root_logger
        get_root_logger().info("User interrupted.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    from traiNNer.check.check_dependencies import check_dependencies
    check_dependencies()
    test_pipeline(str(Path(__file__).resolve().parent))
