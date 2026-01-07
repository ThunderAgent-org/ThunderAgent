import logging
import os
import sys
import traceback

def setup_crash_logging(log_file="crash.log"):
    """
    Sets up a global exception handler to log unhandled exceptions to a file.
    Also captures system exit signals.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        
        # Also print to stderr so it's visible in console logs if available
        print("Uncaught exception:", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

    sys.excepthook = handle_exception
    print(f"Crash logging initialized. Logs will be written to {os.path.abspath(log_file)}")

if __name__ == "__main__":
    setup_crash_logging()

