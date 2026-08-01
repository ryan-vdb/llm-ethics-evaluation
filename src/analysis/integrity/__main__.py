"""Run the integrity analysis with ``python -m src.analysis.integrity``."""


def main() -> None:
    """Import the runner lazily so importing this module has no side effects."""

    from .runner import main as run

    run()


if __name__ == "__main__":
    main()
