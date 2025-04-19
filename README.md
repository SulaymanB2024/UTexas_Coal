# Global Coal Price Forecasting Framework

## Project Overview

This project implements a quantitative framework to explain and forecast global coal prices (specifically targeting [Specify Benchmark, e.g., Newcastle FOB 6000kc NAR]) based on fundamental drivers. It follows the iterative modeling approach detailed in the project plan, progressing from baseline models to time-series econometrics and machine learning ensembles.

The primary goal is to provide reliable 1-3 month ahead forecasts for use in trading strategy development, scenario analysis, and risk management.

## Repository Structure

```
├── config/               # Configuration files (paths, parameters)
│   └── config.yaml
├── data/                 # Project data
│   ├── raw/              # Raw, immutable data downloads
│   ├── interim/          # Intermediate, cleaned data
│   └── processed/        # Final datasets for modeling
├── notebooks/            # Jupyter notebooks for EDA and experimentation
├── reports/              # Generated reports, figures, tables
│   ├── figures/
│   └── tables/
├── scripts/              # Standalone scripts for pipeline execution
├── src/                  # Source code modules
│   ├── data_ingestion/
│   ├── data_processing/
│   ├── modeling/
│   ├── visualization/
│   └── utils/
├── tests/                # Unit and integration tests
├── .gitignore            # Files to ignore in Git
├── LICENSE               # Project license (MIT)
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # Using venv
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    # Or using conda
    # conda create --name coal_forecast python=3.9
    # conda activate coal_forecast
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment Variables (if needed):**
    * Create a `.env` file in the root directory for API keys or other secrets.
    * Example `.env` content:
        ```
        EIA_API_KEY="YOUR_EIA_API_KEY"
        OTHER_SERVICE_CREDENTIALS="..."
        ```

## Usage

* **Data Ingestion:** Run scripts in `scripts/` or relevant functions in `src/data_ingestion/`.
* **EDA:** Explore notebooks in `notebooks/`.
* **Model Training:** Run scripts in `scripts/` or relevant functions in `src/modeling/`.
* **Forecasting:** Run scripts in `scripts/` or relevant functions in `src/modeling/`.

## Contributing

[Optional: Add contribution guidelines if applicable]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
