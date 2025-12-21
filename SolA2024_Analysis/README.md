# Project CICCADA

## Historical data dump format

Historical data is located in `historical_data/` and comprise three key files as described below.

### site_metadata.csv
- site_id: unique site identifer.
- state: site state.
- postcode: site postcode.
- latitude: rounded to nearest ~5km.
- longitude: rounded to nearest ~5km.
- dnsp_name: site DNSP name.
- dc_capacity_kw: site DC capacity in kW (i.e., the total sum of solar panel wattage on site).
- ac_capacity_kw: site AC capacity in kW (i.e., the total sum of inverter output capacity on site).
- export_limit_kw: site export limit in kW, if one exists. Note that this value has been user-entered.
- monitoring_start: monitoring start date on site.
- inverter_info: information about the inverters on site including their installation date, and manufacturer and model.

### circuit_metadata.csv
- site_id: unique site identifier.
- device_id: device identifier.
- device_type: device type (CATCH Power or Watt Watcher).
- circuit_id: circuit identifier.
- circuit_polarity: polarity of the circuit (1 / -1). Should be multiplied by metrology to produce accurate data (e.g., to correct negative PV metrology).
- circuit_type: the type of the circuit, e.g., is it measuring PV, load, hot water, etc. Definitions are as follows:
    - ac_load: Represents Gross AC load monitoring (whole of house consumption)
    - ac_load_net: Represents Net AC load monitoring (whole of house consumption).
    - pv_site: Represents Gross Solar System monitoring
    - pv_site_net: Represents Net Solar System monitoring. These measurements must be included in calculating the sites total household consumption usage if a NET site.
    - pv_inverter: Represents Gross inverter specific Solar System monitoring
    - pv_inverter_net: Represents Net inverter specific Solar System monitoring. These measurements must be included in calculating the sites total household consumption usage if a NET site.
    - weather_sens: Represents weather sensor measurements
    - load_hot_water: Represents load measurements for a hot water system
    - load_hot_water_solar: Represents load measurements for a solar hot water system
    - load_pool: Represents load measurements for a pool
    - load_air_conditioner: Represents load measurements for an air conditioning system
    - load_stove: Represents load measurements for a stove
    - load_lighting: Represents load measurements for the house lighting
    - load_other: Represents measurements for an unspecified load
    - load_ev_charger: Represents measurements for an EV charger
    - battery_storage: Represents measurements for a connected battery
    - load_powerpoint: Represents measurements for a monitored powerpoint(s)
    - load_spa:
    - load_shed: Represents measurements for a shed
    - load_air_compressor: Represents measurements for an air compressor
    - load_refrigerator: Represents measurements for a refrigerator
    - load_laundry: Represents measurements for a laundry
    - load_washer: Represents measurements for a washing machine
    - load_kitchen: Represents measurements for a kitchen
    - load_generator: Represents measurements for a generator
    - load_common_area: Represents measurements for a generic common area
    - load_tenant: Represents measurements for a generic tenant area
    - load_subboard: Represents measurements for a separate subboard
    - load_studio: Represents measurements for a studio
    - load_garage: Represents measurements for a garage
    - load_machine: Represents measurements for a generic machine
    - load_office: Represents measurements for an office
    - load_security: Represents measurements for a security appliance(s)
    - load_other: Represents measurements for an unspecified subcircuit

### metrology

Please note that not all data is present for every circuit, depending on the device and its configuration.

- device_id: device identifier.
- circuit_id: circuit identifier.
- t_stamp: metrology timestampe (UTC).
- power: average power over timestep in W.
- energy: total energy over timestep in Wh.
- energy_reactive: total reactive energy over timestep in Wh.
- energy_import: imported component of energy over timestep in Wh.
- energy_export: exported component of energy over timestep in Wh.
- energy_reactive_import: imported component of reactive energy over timestep in Wh.
- energy_reactive_export: exported component of reactive energy over timestep in Wh.
- power_factor: calculated as energy^2 / (energy^2 + energy_reactive^2).
- voltage: average voltage over timestep.
- current: average current over timestep.

## High-resolution Emergency Backstop Measures (EBM) data

EBM data is located in `ebm_data/` and comprises...TBD