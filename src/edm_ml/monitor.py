import logging


def set_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
