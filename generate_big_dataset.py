#!/usr/bin/env python3
"""
Improved mock data generator for MCO (liquidity risk) prototype
More realistic balance patterns with monthly cycles + archetypes
Light version suitable for laptop development (1 year history)
"""

import argparse
import zipfile
import pandas as pd
import numpy as np
import csv
import io
from datetime import datetime, timedelta
import random

# ── Configuration ───────────────────────────────────────────────────────────────

TARGET_CUSTOMERS        = 3_000                  # Light & fast for development
YEARS_OF_HISTORY        = 1
DAYS_OF_HISTORY         = 365 * YEARS_OF_HISTORY + 35   # ~400 days - small buffer

ENTITIES = [
    ('SG', 'Singapore'),
    ('TH', 'Thailand'),
    ('TW', 'Taiwan')
]

CURRENCIES = ['SGD', 'USD', 'THB', 'TWD']
CURRENCY_WEIGHTS = [5.5, 1.8, 1.4, 0.9]          # Heavier on SGD

# ── Customer Archetypes ─────────────────────────────────────────────────────────
# name, probability, min_balance, max_balance, daily_volatility, avg_txn_per_month
ARCHETYPES = [
    ("young_salary",         0.28,   5_000,   65_000,  0.019,  85),
    ("family_middle",        0.32,  35_000,  320_000, 0.013,  52),
    ("mature_conservative",  0.13, 110_000, 950_000,  0.008,  32),
    ("sme_small_business",   0.11,  55_000, 750_000,  0.024, 140),
    ("foreign_worker",       0.09,   3_500,  85_000,  0.028,  65),
    ("student_young_adult",  0.07,     600,  22_000,  0.038, 125),
]

# FX reference rates (mid 2025 approximate)
FX_BASE_TO_SGD = {
    'SGD': 1.000,
    'USD': 1.335,
    'THB': 0.0395,
    'TWD': 0.0422
}

# ────────────────────────────────────────────────────────────────────────────────

def sample_customer():
    arch_name, _, min_b, max_b, vol, txn_m = random.choices(
        ARCHETYPES, weights=[a[1] for a in ARCHETYPES])[0]

    base_balance = np.random.triangular(min_b, (min_b + max_b) / 1.6, max_b)
    currency = random.choices(CURRENCIES, weights=CURRENCY_WEIGHTS)[0]

    salary_day_options = {
        "young_salary": [25,26,27,28,29,30],
        "family_middle": [24,25,26,27],
        "mature_conservative": [25,27,28],
        "sme_small_business": [25,28,30],
        "foreign_worker": [28,29,30,31],
        "student_young_adult": [20,25,28,30]
    }

    large_outflow_days = {
        "young_salary": [5,10,15],
        "family_middle": [1,5,10],
        "mature_conservative": [8,15,20],
        "sme_small_business": [8,12,18,22],
        "foreign_worker": list(range(1,32)),  # more random
        "student_young_adult": [3,8,13,18,23,28]
    }

    return {
        'archetype': arch_name,
        'base_balance': base_balance,
        'volatility': vol,
        'currency': currency,
        'salary_days': salary_day_options[arch_name],
        'large_outflow_days': large_outflow_days[arch_name],
        'avg_txn_per_month': txn_m
    }


def generate_balance_series(customer, start_date, n_days):
    balance = customer['base_balance']
    balances = []
    current_date = start_date

    for _ in range(n_days):
        day_of_month = current_date.day

        # Monthly salary / main inflow
        if day_of_month in customer['salary_days'] or day_of_month >= 25:
            inflow_factor = np.random.triangular(0.65, 0.95, 1.35)
            inflow = inflow_factor * customer['base_balance'] * 0.38
            balance += inflow

        # Large predictable outflows
        if day_of_month in customer['large_outflow_days']:
            outflow_factor = np.random.triangular(0.45, 0.75, 1.15)
            outflow = outflow_factor * customer['base_balance'] * 0.28
            balance -= outflow

        # Daily random walk
        daily_change = np.random.normal(0, customer['volatility'])
        balance *= (1 + daily_change)

        # Floor at realistic minimum
        balance = max(150.0, balance)

        balances.append(round(balance))  # integer dollars for smaller file size
        current_date += timedelta(days=1)

    return balances


def iter_transactions(customers, start_date, n_days):
    """Yield synthetic transactions per customer with inflows/outflows."""
    txn_id = 1
    outflow_types = ["POS", "TRANSFER_OUT", "BILL_PAY", "CASH_WITHDRAW", "CARD_PAYMENT"]
    inflow_types = ["SALARY", "REFUND", "TRANSFER_IN", "OTHER_IN"]

    for cust in customers:
        base = max(500.0, cust['base_balance'])
        lam_day = max(0.6, cust['avg_txn_per_month'] / 30.0)  # average txns/day
        lam_in = lam_day * 0.35
        lam_out = lam_day * 0.65

        for i in range(n_days):
            dt = start_date + timedelta(days=i)

            # Salary / main inflow (once a month around salary days)
            if dt.day in cust['salary_days'] or (dt.day >= 25 and random.random() < 0.18):
                amount = np.random.triangular(0.25, 0.42, 0.75) * base
                yield [
                    f"T{txn_id:09d}",
                    (dt + timedelta(seconds=random.randint(8*3600, 18*3600))).strftime('%Y-%m-%d %H:%M:%S'),
                    cust['customer_id'],
                    round(amount, 2),
                    cust['currency'],
                    'CR',
                    'SALARY',
                    'EXTERNAL',
                    cust['entity_code'],
                ]
                txn_id += 1

            # Regular inflows
            n_in = np.random.poisson(lam_in)
            for _ in range(n_in):
                amount = np.random.lognormal(mean=np.log(base * 0.006), sigma=0.8)
                yield [
                    f"T{txn_id:09d}",
                    (dt + timedelta(seconds=random.randint(7*3600, 21*3600))).strftime('%Y-%m-%d %H:%M:%S'),
                    cust['customer_id'],
                    round(max(10.0, amount), 2),
                    cust['currency'],
                    'CR',
                    random.choice(inflow_types),
                    'EXTERNAL',
                    cust['entity_code'],
                ]
                txn_id += 1

            # Regular outflows
            n_out = np.random.poisson(lam_out)
            for _ in range(n_out):
                amount = np.random.lognormal(mean=np.log(base * 0.004), sigma=0.9)
                yield [
                    f"T{txn_id:09d}",
                    (dt + timedelta(seconds=random.randint(6*3600, 22*3600))).strftime('%Y-%m-%d %H:%M:%S'),
                    cust['customer_id'],
                    round(max(8.0, amount), 2),
                    cust['currency'],
                    'DR',
                    random.choice(outflow_types),
                    'EXTERNAL',
                    cust['entity_code'],
                ]
                txn_id += 1

            # Occasional large outflow
            if dt.day in cust['large_outflow_days'] and random.random() < 0.55:
                amount = np.random.triangular(0.18, 0.32, 0.65) * base
                yield [
                    f"T{txn_id:09d}",
                    (dt + timedelta(seconds=random.randint(9*3600, 17*3600))).strftime('%Y-%m-%d %H:%M:%S'),
                    cust['customer_id'],
                    round(amount, 2),
                    cust['currency'],
                    'DR',
                    'LARGE_TRANSFER',
                    'EXTERNAL',
                    cust['entity_code'],
                ]
                txn_id += 1


def generate_big_dataset(output_zip='mco_mock_data_big.zip'):
    print("Generating realistic MCO mock dataset (light version - 1 year)...")
    print(f"  → Target customers:   {TARGET_CUSTOMERS:,}")
    print(f"  → History:            {DAYS_OF_HISTORY} days "
          f"({YEARS_OF_HISTORY} year + {DAYS_OF_HISTORY - 365*YEARS_OF_HISTORY} days buffer)")
    print(f"  → Expected rows:      ~{TARGET_CUSTOMERS * DAYS_OF_HISTORY:,}\n")

    np.random.seed(42)
    random.seed(42)

    start_date = datetime(2024, 1, 1)   # Starting point - feel free to adjust

    # ── Generate customers ──────────────────────────────────────────────────────
    customers = []
    customer_id_counter = 1

    print("  Generating customer master...")
    for _ in range(TARGET_CUSTOMERS):
        params = sample_customer()
        entity = random.choice(ENTITIES)

        cust_id = f"C{customer_id_counter:07d}"
        customer_id_counter += 1

        customers.append({
            'customer_id': cust_id,
            'entity_code': entity[0],
            'entity_name': entity[1],
            'currency': params['currency'],
            'archetype': params['archetype'],
            'base_balance': params['base_balance'],
            'volatility': params['volatility'],
            'salary_days': params['salary_days'],
            'large_outflow_days': params['large_outflow_days'],
            'avg_txn_per_month': params['avg_txn_per_month'],
            'onboard_date': (start_date + timedelta(days=random.randint(-180, 90))).strftime('%Y-%m-%d'),
            'residency_status': 'Resident' if random.random() < 0.82 else 'NonResident',
            'customer_type': 'Retail' if random.random() < 0.84 else 'SME',
            'segment': 'SME' if params['archetype'] == 'sme_small_business' else 'Retail',
        })

    # ── Write to ZIP ────────────────────────────────────────────────────────────
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z_out:

        # 1. CUSTOMER_MASTER
        print("  → Writing CUSTOMER_MASTER.csv ...")
        with z_out.open('CUSTOMER_MASTER.csv', 'w') as f_bin:
            with io.TextIOWrapper(f_bin, encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['customer_id', 'customer_type', 'segment', 'onboard_date',
                               'residency_status', 'entity_code', 'entity_name', 'currency'])
                for row in customers:
                    writer.writerow([
                        row['customer_id'],
                        row['customer_type'],
                        row['segment'],
                        row['onboard_date'],
                        row['residency_status'],
                        row['entity_code'],
                        row['entity_name'],
                        row['currency']
                    ])

        # 2. BALANCE_DAILY - streaming
        print("  → Writing BALANCE_DAILY.csv ...")
        with z_out.open('BALANCE_DAILY.csv', 'w') as f_bin:
            with io.TextIOWrapper(f_bin, encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['customer_id', 'as_of_date', 'entity_code', 'currency', 'end_of_day_balance'])

                date_strings = []
                current_date = start_date
                for _ in range(DAYS_OF_HISTORY):
                    date_strings.append(current_date.strftime('%Y-%m-%d'))
                    current_date += timedelta(days=1)

                for cust in customers:
                    balances = generate_balance_series(cust, start_date, DAYS_OF_HISTORY)

                    for dt_str, bal in zip(date_strings, balances):
                        writer.writerow([
                            cust['customer_id'],
                            dt_str,
                            cust['entity_code'],
                            cust['currency'],
                            bal
                        ])

        # 3. FX_RATES_DAILY
        print("  → Writing FX_RATES_DAILY.csv ...")
        with z_out.open('FX_RATES_DAILY.csv', 'w') as f_bin:
            with io.TextIOWrapper(f_bin, encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'from_currency', 'to_currency', 'fx_rate'])

                current_date = start_date
                for _ in range(DAYS_OF_HISTORY):
                    d_str = current_date.strftime('%Y-%m-%d')
                    for fc in CURRENCIES:
                        for tc in CURRENCIES:
                            if fc == tc:
                                rate = 1.000000
                            else:
                                base = FX_BASE_TO_SGD[tc] / FX_BASE_TO_SGD[fc]
                                noise = np.random.normal(0, 0.004)
                                rate = max(0.01, round(base * (1 + noise), 6))
                            writer.writerow([d_str, fc, tc, f"{rate:.6f}"])
                    current_date += timedelta(days=1)

        # 4. TXN_LEDGER - synthetic inflows/outflows
        print("  → Writing TXN_LEDGER.csv ...")
        with z_out.open('TXN_LEDGER.csv', 'w') as f_bin:
            with io.TextIOWrapper(f_bin, encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['txn_id','txn_datetime','customer_id','amount','currency',
                               'dr_cr_flag','txn_type_code','counterparty_type','entity_code'])

                for row in iter_transactions(customers, start_date, DAYS_OF_HISTORY):
                    writer.writerow(row)

    print(f"\nGeneration completed!")
    print(f"   Output file:           {output_zip}")
    print(f"   Customers:             {len(customers):,}")
    print(f"   Days of history:       {DAYS_OF_HISTORY}")
    print(f"   Balance rows (approx): {len(customers) * DAYS_OF_HISTORY:,}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate 1-year mock MCO dataset (light version)")
    parser.add_argument("--out", default="mco_mock_data_big.zip", help="Output ZIP filename")
    args = parser.parse_args()

    generate_big_dataset(output_zip=args.out)