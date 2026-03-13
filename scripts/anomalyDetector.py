import argparse
import funcs as f
import os
import sys
import time
import logging
from datetime import datetime


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logfile = os.path.join(log_dir, f"run_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"Logging to: {logfile}")
    return logfile

def main():
    logfile = setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True, help='Input Url')
    parser.add_argument('--outputFile', required=True, help='Output parquet file')
    parser.add_argument('--pulse_output', required=True, help='Output pulse info parquet file')
    parser.add_argument('--model_path', required = True, help = 'Model used')

    args = parser.parse_args()

    logging.info("---- PIPELINE START ----")
    logging.info(f"input url: {args.url}")
    logging.info(f"output metrics: {args.outputFile}")
    logging.info(f"output pulses: {args.pulse_output}")
    logging.info(f"model: {args.model_path}")

    t0 = time.time()
    try:
        f.rl(url=args.url, outputFile=args.outputFile, pulse_output=args.pulse_output, model_path=args.model_path)
        logging.info(f"PIPELINE FINISHED OK in {time.time()-t0:.1f}s")
    except Exception:
        logging.exception("PIPELINE FAILED with exception:")
        raise

if __name__ == "__main__":
    main()
