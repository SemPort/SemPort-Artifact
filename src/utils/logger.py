import logging
import sys

class ColorFormatter(logging.Formatter):

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }

    RESET = "\033[0m"

    def format(self, record):
        level = record.levelname
        if level in self.COLORS:
            color = self.COLORS[level]

            record.levelname = f"{color}{record.levelname}{self.RESET}"
            record.name = f"{color}{record.name}{self.RESET}"
            record.funcName = f"{color}{record.funcName}{self.RESET}"
            record.msg = f"{color}{record.msg}{self.RESET}"

        return super().format(record)

def get_logger(name, level=logging.INFO, log_path=None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s | %(message)s"

        logger.setLevel(level)

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(ColorFormatter(formatter))
        logger.addHandler(console)

#         logger.debug(f'[+] Set logger for {name}, log_path: {log_path}.')
    return logger
