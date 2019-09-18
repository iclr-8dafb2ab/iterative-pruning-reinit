import argparse
import logging


def setup_logging(logfile="", debug=False):
    logger = logging.getLogger()

    # add color and structure
    logging.basicConfig(
        format="%(levelname)-8s\033[1m%(name)-8s%(process)d\033[0m: %(message)s"
    )
    logging.addLevelName(
        logging.WARNING,
        "\033[1;31m{:8}\033[1;0m".format(
            logging.getLevelName(logging.WARNING)
        ),
    )
    logging.addLevelName(
        logging.ERROR,
        "\033[1;35m{:8}\033[1;0m".format(logging.getLevelName(logging.ERROR)),
    )
    logging.addLevelName(
        logging.INFO,
        "\033[1;32m{:8}\033[1;0m".format(logging.getLevelName(logging.INFO)),
    )
    logging.addLevelName(
        logging.DEBUG,
        "\033[1;34m{:8}\033[1;0m".format(logging.getLevelName(logging.DEBUG)),
    )

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # if output file provided, log to there as well
    if logfile != "":
        handler = logging.FileHandler(logfile, mode="w")
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger
