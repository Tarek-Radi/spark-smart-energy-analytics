"""
generate_data.py

Project: Smart Energy Consumption Analytics with Apache Spark
Author:  Senior Data Engineer (IoT / Smart Buildings)

Generates a realistic synthetic dataset simulating 15-minute electricity
readings collected by smart meters / IoT devices across multiple commercial
buildings over several months.

The generator is fully vectorized with pandas/numpy for performance and
produces ~100,000 records after down-sampling a much denser raw time grid
(mirroring the fact that real IoT fleets rarely report every device at
every interval).

Output:
    data/raw/synthetic_energy_data.csv

Run:
    python generate_data.py
"""

import random
from datetime import datetime

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. CONSTANTS & CONFIGURATION
# --------------------------------------------------------------------------

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_PATH = "data/raw/synthetic_energy_data.csv"

# Time range: 4 months of 15-minute readings
START_DATE = datetime(2024, 1, 1, 0, 0, 0)
END_DATE = datetime(2024, 4, 30, 23, 45, 0)
FREQ = "15min"

# Buildings
BUILDING_IDS = [f"B{str(i).zfill(3)}" for i in range(1, 11)]  # B001..B010

# Devices per building (mix of equipment types installed in each building)
DEVICES_PER_BUILDING = 5

# Final target size of the dataset (after sampling + injected anomalies)
TARGET_RECORDS = 100_000

# Device types and their realistic energy consumption ranges (kWh / 15 min)
DEVICE_ENERGY_RANGES = {
    "Lighting": (0.1, 3.0),
    "HVAC": (5.0, 25.0),
    "Elevator": (0.5, 4.0),
    "Server": (3.0, 15.0),
    "Production Machine": (20.0, 80.0),
    "Pump": (2.0, 10.0),
    "Air Compressor": (8.0, 30.0),
    "Cooling System": (6.0, 25.0),
}

# Plausible department(s) for each device type
DEVICE_TYPE_DEPARTMENTS = {
    "Lighting": ["Finance", "HR", "Administration", "Production", "Warehouse", "IT"],
    "HVAC": ["Finance", "HR", "Administration", "IT", "Production"],
    "Elevator": ["Administration"],
    "Server": ["IT"],
    "Production Machine": ["Production"],
    "Pump": ["Warehouse", "Production"],
    "Air Compressor": ["Production", "Warehouse"],
    "Cooling System": ["IT", "Production"],
}

DEVICE_STATUSES = ["Running", "Idle", "Maintenance", "Offline"]

# Anomaly / data-quality injection rates
SPIKE_EVENT_COUNT = 250            # number of high-consumption spike events
OUTAGE_EVENT_COUNT = 150           # number of device outage events
DUPLICATE_RATE = 0.01              # ~1% duplicate records
MISSING_VALUE_RATE = 0.02          # ~2% missing (null) values in numeric cols
MISSING_TIMESTAMP_RATE = 0.015     # ~1.5% of rows dropped entirely
INVALID_VALUE_RATE = 0.01          # ~1% rows with corrupted/invalid values

NUMERIC_COLUMNS_FOR_NULLS = ["energy_kwh", "voltage", "current", "temperature"]


# --------------------------------------------------------------------------
# 2. DEVICE & BUILDING SETUP
# --------------------------------------------------------------------------

def generate_building_profiles():
    """Assign each building a random consumption multiplier so that some
    buildings are systematically heavier consumers than others."""
    return {
        b_id: round(np.random.uniform(0.7, 1.6), 2) for b_id in BUILDING_IDS
    }


def generate_devices():
    """Create a fixed fleet of devices, each permanently tied to one
    building, one device type, and one department."""
    device_types = list(DEVICE_ENERGY_RANGES.keys())
    devices = []
    device_counter = 1

    for building_id in BUILDING_IDS:
        for _ in range(DEVICES_PER_BUILDING):
            device_type = random.choice(device_types)
            department = random.choice(DEVICE_TYPE_DEPARTMENTS[device_type])
            device_id = f"DEV-{str(device_counter).zfill(4)}"
            devices.append(
                {
                    "device_id": device_id,
                    "building_id": building_id,
                    "device_type": device_type,
                    "department": department,
                }
            )
            device_counter += 1

    return pd.DataFrame(devices)


# --------------------------------------------------------------------------
# 3. BASE TIME GRID
# --------------------------------------------------------------------------

def build_base_grid(devices_df):
    """Cross-join every device with every timestamp in the study period."""
    timestamps = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQ)
    ts_df = pd.DataFrame({"timestamp": timestamps})

    devices_df["_key"] = 1
    ts_df["_key"] = 1
    grid = devices_df.merge(ts_df, on="_key").drop(columns="_key")
    devices_df.drop(columns="_key", inplace=True)

    return grid


# --------------------------------------------------------------------------
# 4. TIME-BASED FEATURES
# --------------------------------------------------------------------------

def add_time_features(df):
    """Attach calendar features used to model daily/weekly/seasonal cycles."""
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Mon ... 6=Sun
    df["is_weekend"] = df["day_of_week"] >= 5
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    return df


def compute_temperature(df):
    """Simulate outdoor/ambient temperature with seasonal + daily patterns."""
    # Seasonal component: sinusoid peaking mid-summer (Jan start -> cooler)
    seasonal = 18 + 8 * np.sin(2 * np.pi * (df["day_of_year"] - 30) / 365)

    # Daily component: warmer in the afternoon, cooler at night
    daily = 5 * np.sin(2 * np.pi * (df["hour"] - 6) / 24)

    noise = np.random.normal(0, 1.2, size=len(df))

    df["temperature"] = (seasonal + daily + noise).round(2)
    return df


def business_hours_factor(hour, is_weekend, device_type):
    """Return a multiplier reflecting typical working-hour demand curves."""
    working_hours = (hour >= 8) & (hour < 19)

    if device_type in ("Server",):
        # Servers/data-center loads stay roughly constant day and night
        return np.full(len(hour), 1.0)

    if device_type in ("HVAC", "Cooling System"):
        # Climate control still runs off-hours but at reduced levels
        factor = np.where(working_hours, 1.0, 0.45)
        factor = np.where(is_weekend, factor * 0.7, factor)
        return factor

    # Lighting, Elevator, Production Machine, Pump, Air Compressor
    factor = np.where(working_hours, 1.0, 0.12)
    weekend_penalty = np.where(is_weekend, 0.35, 1.0)
    factor = factor * weekend_penalty
    return factor


# --------------------------------------------------------------------------
# 5. STATUS ASSIGNMENT
# --------------------------------------------------------------------------

def assign_device_status(df):
    """Assign a device_status per reading, mostly driven by working hours."""
    working_hours = (df["hour"] >= 8) & (df["hour"] < 19) & (~df["is_weekend"])

    status = np.empty(len(df), dtype=object)

    # Base probabilities during working hours vs off hours
    rand = np.random.random(len(df))

    on_hours_probs = [0.90, 0.06, 0.02, 0.02]     # Running, Idle, Maint, Offline
    off_hours_probs = [0.35, 0.50, 0.05, 0.10]

    cum_on = np.cumsum(on_hours_probs)
    cum_off = np.cumsum(off_hours_probs)

    for i, statuses_cum in [(True, cum_on), (False, cum_off)]:
        mask = working_hours if i else ~working_hours
        r = rand[mask.values]
        s = np.select(
            [r < statuses_cum[0], r < statuses_cum[1], r < statuses_cum[2]],
            ["Running", "Idle", "Maintenance"],
            default="Offline",
        )
        status[mask.values] = s

    df["device_status"] = status
    return df


# --------------------------------------------------------------------------
# 6. ENERGY, VOLTAGE, CURRENT
# --------------------------------------------------------------------------

def compute_energy(df, building_multipliers):
    """Compute realistic energy_kwh per reading based on device type,
    time-of-day/week factors, temperature (for HVAC/Cooling), building
    multiplier, and current device status."""

    energy = np.zeros(len(df))

    building_mult = df["building_id"].map(building_multipliers).values

    for device_type, (low, high) in DEVICE_ENERGY_RANGES.items():
        mask = (df["device_type"] == device_type).values
        if not mask.any():
            continue

        base = np.random.uniform(low, high, size=mask.sum())

        hour_vals = df.loc[mask, "hour"].values
        weekend_vals = df.loc[mask, "is_weekend"].values
        tod_factor = business_hours_factor(hour_vals, weekend_vals, device_type)

        # Temperature effect: HVAC / Cooling Systems scale with deviation
        # from a comfortable 21C setpoint (hotter or colder -> more energy)
        if device_type in ("HVAC", "Cooling System"):
            temp_dev = np.abs(df.loc[mask, "temperature"].values - 21)
            temp_factor = 1 + (temp_dev / 15)
        else:
            temp_factor = 1.0

        noise = np.random.normal(1.0, 0.08, size=mask.sum())

        values = base * tod_factor * temp_factor * building_mult[mask] * noise
        energy[mask] = values

    df["energy_kwh"] = energy

    # Status overrides: Maintenance/Offline consume (near) zero energy
    df.loc[df["device_status"] == "Offline", "energy_kwh"] = np.round(
        np.random.uniform(0, 0.05, size=(df["device_status"] == "Offline").sum()), 3
    )
    df.loc[df["device_status"] == "Maintenance", "energy_kwh"] = np.round(
        np.random.uniform(0, 0.3, size=(df["device_status"] == "Maintenance").sum()), 3
    )

    df["energy_kwh"] = df["energy_kwh"].round(3)
    return df


def compute_electrical(df):
    """Derive voltage (normally distributed ~220V) and current from the
    computed energy consumption (P = V * I -> I = P / V)."""
    df["voltage"] = np.round(np.random.normal(220, 4, size=len(df)), 2)

    # Average power (Watts) drawn during the 15-minute interval
    power_w = (df["energy_kwh"] / 0.25) * 1000
    current = power_w / df["voltage"]
    current = current * np.random.normal(1.0, 0.05, size=len(df))  # sensor noise
    df["current"] = np.round(current.clip(lower=0), 3)

    return df


# --------------------------------------------------------------------------
# 7. ANOMALY INJECTION
# --------------------------------------------------------------------------

def inject_high_consumption_spikes(df):
    """Randomly select devices and short time windows (30-120 min) where
    energy usage spikes sharply above normal levels."""
    device_ids = df["device_id"].unique()

    for _ in range(SPIKE_EVENT_COUNT):
        device_id = random.choice(device_ids)
        device_rows = df.index[df["device_id"] == device_id]
        if len(device_rows) < 8:
            continue

        start_idx = random.choice(device_rows[:-8])
        duration_steps = random.randint(2, 8)  # 30-120 minutes (15-min steps)
        window = df.loc[start_idx: start_idx + duration_steps - 1]
        window_idx = window[window["device_id"] == device_id].index

        spike_multiplier = np.random.uniform(2.0, 5.0)
        df.loc[window_idx, "energy_kwh"] = (
            df.loc[window_idx, "energy_kwh"] * spike_multiplier
        ).round(3)

    return df


def inject_device_outages(df):
    """Randomly select devices and contiguous periods where they go
    fully offline (status=Offline, energy/current ~0)."""
    device_ids = df["device_id"].unique()

    for _ in range(OUTAGE_EVENT_COUNT):
        device_id = random.choice(device_ids)
        device_rows = df.index[df["device_id"] == device_id]
        if len(device_rows) < 20:
            continue

        start_idx = random.choice(device_rows[:-20])
        duration_steps = random.randint(4, 16)  # 1-4 hours
        window = df.loc[start_idx: start_idx + duration_steps - 1]
        window_idx = window[window["device_id"] == device_id].index

        df.loc[window_idx, "device_status"] = "Offline"
        df.loc[window_idx, "energy_kwh"] = 0.0
        df.loc[window_idx, "current"] = 0.0

    return df


def inject_invalid_values(df):
    """Corrupt a small percentage of rows with physically invalid readings
    (negative energy, out-of-range voltage/current/temperature)."""
    n = len(df)
    n_invalid = int(n * INVALID_VALUE_RATE)
    idx = np.random.choice(df.index, size=n_invalid, replace=False)
    chunks = np.array_split(idx, 5)

    # Negative energy consumption
    df.loc[chunks[0], "energy_kwh"] = -np.abs(
        np.random.uniform(1, 20, size=len(chunks[0]))
    ).round(3)

    # Voltage below safe threshold
    df.loc[chunks[1], "voltage"] = np.random.uniform(80, 149, size=len(chunks[1])).round(2)

    # Voltage above safe threshold
    df.loc[chunks[2], "voltage"] = np.random.uniform(281, 350, size=len(chunks[2])).round(2)

    # Negative current
    df.loc[chunks[3], "current"] = -np.abs(
        np.random.uniform(1, 15, size=len(chunks[3]))
    ).round(3)

    # Unrealistic temperature (sensor fault)
    extreme_cold = np.random.uniform(-40, -20, size=len(chunks[4]) // 2)
    extreme_hot = np.random.uniform(60, 90, size=len(chunks[4]) - len(extreme_cold))
    df.loc[chunks[4], "temperature"] = np.round(
        np.concatenate([extreme_cold, extreme_hot]), 2
    )

    return df


# --------------------------------------------------------------------------
# 8. DATA QUALITY ISSUES
# --------------------------------------------------------------------------

def inject_missing_values(df):
    """Randomly null out ~2% of numeric readings across selected columns."""
    for col in NUMERIC_COLUMNS_FOR_NULLS:
        n_missing = int(len(df) * (MISSING_VALUE_RATE / len(NUMERIC_COLUMNS_FOR_NULLS)))
        idx = np.random.choice(df.index, size=n_missing, replace=False)
        df.loc[idx, col] = np.nan
    return df


def inject_missing_timestamps(df):
    """Simulate sensor dropout by removing a random subset of rows
    entirely (missing readings)."""
    n_drop = int(len(df) * MISSING_TIMESTAMP_RATE)
    drop_idx = np.random.choice(df.index, size=n_drop, replace=False)
    return df.drop(index=drop_idx).reset_index(drop=True)


def inject_duplicates(df):
    """Append a random sample of existing rows back into the dataset to
    simulate duplicate transmissions from IoT gateways."""
    n_dupes = int(len(df) * DUPLICATE_RATE)
    dupes = df.sample(n=n_dupes, random_state=RANDOM_SEED, replace=True)
    return pd.concat([df, dupes], ignore_index=True)


# --------------------------------------------------------------------------
# 9. SAMPLING TO TARGET SIZE
# --------------------------------------------------------------------------

def sample_to_target(df, target=TARGET_RECORDS):
    """Down-sample the dense base grid to roughly the target record count,
    mimicking real fleets where not every device reports every interval."""
    if len(df) <= target:
        return df
    return df.sample(n=target, random_state=RANDOM_SEED).reset_index(drop=True)


# --------------------------------------------------------------------------
# 10. MAIN PIPELINE
# --------------------------------------------------------------------------

def main():
    print("Generating building profiles and device fleet...")
    building_multipliers = generate_building_profiles()
    devices_df = generate_devices()

    print(f"Building base time grid ({len(devices_df)} devices)...")
    df = build_base_grid(devices_df)
    df = add_time_features(df)

    print("Computing temperature and device status...")
    df = compute_temperature(df)
    df = assign_device_status(df)

    print("Computing energy, voltage and current...")
    df = compute_energy(df, building_multipliers)
    df = compute_electrical(df)

    print(f"Down-sampling dense grid ({len(df)} rows) to ~{TARGET_RECORDS} records...")
    df = sample_to_target(df, TARGET_RECORDS)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Injecting anomalies (spikes, outages)...")
    df = inject_high_consumption_spikes(df)
    df = inject_device_outages(df)

    print("Injecting invalid sensor values...")
    df = inject_invalid_values(df)

    print("Injecting data quality issues (nulls, duplicates, missing timestamps)...")
    df = inject_missing_values(df)
    df = inject_duplicates(df)
    df = inject_missing_timestamps(df)

    # Final column ordering (drop internal helper columns)
    final_columns = [
        "timestamp",
        "building_id",
        "department",
        "device_id",
        "device_type",
        "energy_kwh",
        "voltage",
        "current",
        "temperature",
        "device_status",
    ]
    df = df[final_columns]

    # Shuffle rows so duplicates/nulls aren't clustered, then keep a stable
    # secondary sort by timestamp for readability
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Final dataset shape: {df.shape}")

    import os
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
