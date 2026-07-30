# Spark Smart Energy Analytics

## Overview

**Spark Smart Energy Analytics** is a practical Data Engineering project built with **Apache Spark** and **PySpark** to analyze electricity consumption across multiple buildings, departments, and devices.

The project focuses on building an end-to-end batch analytics workflow that includes:

- Synthetic data generation
- Data ingestion with Apache Spark
- Basic RDD operations for learning purposes
- Data cleaning and validation with Spark DataFrames
- Missing and duplicated reading handling
- Energy consumption analysis
- Anomaly and spike detection
- Device outage detection
- Missing reading period detection
- Ranking and running totals with Window Functions
- Aggregations using `rollup` and `cube`
- Organized Parquet outputs

This repository is designed as a hands-on learning project and a portfolio-ready demonstration of Spark-based data engineering skills.

---

## Project Objectives

The main objectives are to:

1. Generate a realistic synthetic smart energy dataset.
2. Read and inspect the dataset using Apache Spark.
3. Practice essential RDD transformations and actions.
4. Clean invalid, duplicated, and missing records.
5. Analyze energy consumption by building, department, device, and device type.
6. Detect unusual energy consumption spikes.
7. Identify device outages and missing reading periods.
8. Apply Spark Window Functions such as:
   - `lag`
   - `lead`
   - `rank`
   - `dense_rank`
   - `row_number`
   - Running totals
9. Use `rollup` and `cube` for multidimensional aggregations.
10. Store processed datasets and analytical outputs in Parquet format.

---

## Dataset Schema

The synthetic dataset will include the following columns:

| Column | Description |
|---|---|
| `timestamp` | Date and time of the energy reading |
| `building_id` | Unique building identifier |
| `department` | Department using the device |
| `device_id` | Unique device identifier |
| `device_type` | Type of electrical device |
| `energy_consumption_kwh` | Energy consumed in kilowatt-hours |
| `voltage` | Device voltage reading |
| `current` | Device electrical current |
| `temperature` | Device or environment temperature |
| `device_status` | Device operational status |

The generated data will intentionally include:

- Missing values
- Duplicate records
- Invalid measurements
- Consumption spikes
- Device outage periods
- Missing reading intervals

These cases will be used to practice realistic data cleaning and monitoring workflows.

---

## Planned Project Structure

```text
spark-smart-energy-analytics/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── generate_data.py
│   ├── clean_energy_data.py
│   ├── energy_analytics.py
│   ├── anomaly_detection.py
│   └── outage_detection.py
├── tests/
├── output/
│   ├── daily_building_consumption/
│   ├── department_consumption_summary/
│   ├── device_consumption_ranking/
│   ├── high_consumption_alerts/
│   ├── energy_consumption_anomalies/
│   ├── device_outage_periods/
│   └── missing_reading_periods/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Planned Outputs

The pipeline will generate the following analytical datasets:

- **Daily Building Consumption**
- **Department Consumption Summary**
- **Device Consumption Ranking**
- **High Consumption Alerts**
- **Energy Consumption Anomalies**
- **Device Outage Periods**
- **Missing Reading Periods**

All final outputs will be stored in Parquet format.

---

## Technologies

- Python
- Apache Spark
- PySpark
- Spark SQL
- Spark DataFrames
- Spark RDDs
- Parquet
- Git
- GitHub

---

## Learning Focus

The project is divided into practical stages that connect directly to important Apache Spark concepts.

### Core Spark Topics

- Spark architecture basics
- Driver and executors
- Transformations and actions
- Lazy evaluation
- RDD fundamentals
- DataFrame operations
- Spark SQL functions
- Schema enforcement
- Data cleaning
- Aggregations
- Window Functions
- Partitions
- Shuffle operations
- Parquet storage

RDDs will be used only for learning and comparison. Spark DataFrames will be the primary interface throughout the project.

---

## Six-Day Implementation Plan

### Day 1 — Project Setup and Data Generation

- Create the repository structure
- Configure the Python environment
- Generate a realistic synthetic dataset
- Introduce controlled data quality issues

### Day 2 — Spark Fundamentals and Data Ingestion

- Create a Spark session
- Read the raw dataset
- Inspect schema and partitions
- Practice basic RDD transformations and actions

### Day 3 — Data Cleaning

- Remove duplicates
- Handle missing values
- Validate numerical readings
- Standardize categorical values
- Save cleaned data as Parquet

### Day 4 — Energy Consumption Analytics

- Analyze consumption by building
- Analyze consumption by department
- Analyze consumption by device type
- Use grouping, aggregation, `rollup`, and `cube`

### Day 5 — Window Functions and Detection Logic

- Rank buildings and devices
- Calculate running totals
- Use `lag` and `lead`
- Detect consumption spikes
- Detect outages and missing reading periods

### Day 6 — Testing, Documentation, and Portfolio Preparation

- Validate analytical outputs
- Add basic tests
- Improve project structure
- Add an architecture diagram
- Complete the README
- Prepare sample outputs for the portfolio

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-github-username>/spark-smart-energy-analytics.git
cd spark-smart-energy-analytics
```

### 2. Create a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the Synthetic Dataset

```bash
python src/generate_data.py
```

Additional execution commands will be added as each pipeline stage is implemented.

---

## Current Status

The project is currently in the initial setup phase.

- [Done] Create repository structure
- [Done] Add project Requirments
- [ ] Generate synthetic energy data
- [ ] Build Spark ingestion workflow
- [ ] Add RDD learning examples
- [ ] Clean and validate the dataset
- [ ] Build energy analytics outputs
- [ ] Detect anomalies
- [ ] Detect outages
- [ ] Detect missing reading periods
- [ ] Add tests
- [ ] Add architecture diagram
- [ ] Finalize documentation

---

## Engineering Guidelines

The project will follow these principles:

- Use clear and consistent English names for files, variables, functions, and commits.
- Use Spark DataFrames as the primary processing API.
- Use RDDs only for educational comparison.
- Define schemas explicitly when possible.
- Avoid unnecessary shuffle operations.
- Inspect and manage partitions carefully.
- Separate ingestion, cleaning, analytics, and detection logic.
- Save intermediate and final datasets in organized directories.
- Keep the project simple before adding advanced optimizations.
- Document both correct solutions and more performance-efficient alternatives.

---

## Future Improvements

Possible improvements after completing the core version:

- Add automated unit tests with `pytest`
- Add configurable pipeline parameters
- Add structured logging
- Add data quality reports
- Add a Docker environment
- Add orchestration with Apache Airflow
- Add cloud storage integration
- Add a dashboard for visualizing results

These improvements will only be considered after the core Spark concepts are fully implemented.

---

## Author

**Tarek Radi**

Data Engineering learner building practical portfolio projects with Python, SQL, Apache Spark, and cloud technologies.

---

## License

This project is intended for educational and portfolio purposes.