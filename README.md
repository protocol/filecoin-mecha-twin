# Filecoin Mechanical Twin

Mechanistic model for the Filecoin Economy. You can use this model to forecast all the components underlying Filecoin's circulating supply (i.e., minting, vesting, locking and burning), based on simple assumption about future storage provider behavior.


## Prerequisites

```
numpy>=1.23.1
pandas>=1.4.3
requests>=2.28.1
matplotlib>=3.5.2
seaborn>=0.11.2
```

## Installing

This package is available on PyPI and thus can be directly installed with pip:

```
pip install mechaFIL
```

Alternatively, this package can be installed from source by cloning this repository and installing it manually with the command:

```
python setup.py install
```

## Usage

There is a high-level function that can be used to run a forecast/simulation of circulating supply and its components. First you need to import the relevant packages:

```python
import mechafil
import datetime
```

Now, you need to define some parameters:

```python
# Starting date for the simulation
start_date = datetime.date(2021, 3, 15)
# Current date
current_date = datetime.date(2022, 11, 1) 
# Number of days to run the simulation (after current_rate)
forecast_length = 365
# Renewal rate of all sectors (percentage of raw-byte that will renew)
renewal_rate= 0.6
# Raw-byte power (in PiB) that is onboarded every day
rb_onboard_power = 12.0
# Percentage of raw-byte power onboarded that contains FIL+ deals
fil_plus_rate = 0.098
# Sector duration of newly onboarding sectors
duration = 360
```

An important note regarding the dates - due to data availability, the start date cannot be earlier than 2021-03-15!

Now, you can call the simulation function and collect the data in a DataFrame:

```python
cil_df = mechafil.run_simple_sim(start_date,
    current_date,
    forecast_length,
    renewal_rate,
    rb_onboard_power,
    fil_plus_rate,
    duration)

cil_df.head()
```

You can also run part of the simulation separately. To see more examples, check the available [notebooks](https://github.com/protocol/filecoin-mecha-twin/tree/main/notebooks).

## References

* [Power model spec](https://hackmd.io/@cryptoecon/SkapZkrdc)
* [Locking model spec](https://hackmd.io/@cryptoecon/SJv_CGvY9)