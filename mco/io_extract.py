# mco/io_extract.py
"""
I/O utilities:
- unzip dataset ZIP into a folder
- load CSV tables into pandas

Expected ZIP contents:
- CUSTOMER_MASTER.csv
- FX_RATES_DAILY.csv
- TXN_LEDGER.csv
- BALANCE_DAILY.csv
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def unzip_to_dir(zip_path: Path, out_dir: Path) -> Path:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
        logger.info(f"Extracted {zip_path} to {out_dir}")
        return out_dir
    except FileNotFoundError:
        logger.error(f"ZIP file not found: {zip_path}")
        raise
    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP file: {zip_path}")
        raise


def load_tables(data_dir: Path):
    try:
        cust = pd.read_csv(data_dir / "CUSTOMER_MASTER.csv", parse_dates=["onboard_date"])
        fx = pd.read_csv(data_dir / "FX_RATES_DAILY.csv", parse_dates=["date"])
        txn = pd.read_csv(data_dir / "TXN_LEDGER.csv", parse_dates=["txn_datetime"])
        bal = pd.read_csv(data_dir / "BALANCE_DAILY.csv", parse_dates=["as_of_date"])
        logger.info(f"Loaded {len(cust)} customers, {len(txn)} transactions, {len(bal)} balance records")
        if len(cust) == 0 or len(txn) == 0 or len(bal) == 0:
            logger.warning("One or more tables are empty")
        return cust, fx, txn, bal
    except FileNotFoundError as e:
        logger.error(f"Required CSV file not found in {data_dir}: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV files: {e}")
        raise
