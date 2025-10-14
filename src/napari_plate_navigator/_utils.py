import functools
import logging


def log_method(func):
    """Decorator: Auto-logs entry/exit with class.method."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        logger = logging.getLogger(
            __name__
        )  # Always fetch here—no default/conditional
        class_name = self.__class__.__name__
        method_name = func.__name__
        logger.info("Entering %s.%s", class_name, method_name)
        try:
            result = func(self, *args, **kwargs)
            logger.info("Exiting %s.%s", class_name, method_name)
            return result
        except Exception as e:
            logger.info("Error in %s.%s: %s", class_name, method_name, e)
            raise

    return wrapper
