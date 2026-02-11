# CS506-Final-Project
A Geospatial Equity Analysis of Boston Bus Performance
[CS 506 Project Proposal.docx](https://github.com/user-attachments/files/25241103/CS.506.Project.Proposal.docx)
# MBTA Bus Reliability & Equity Analysis

## 1. Project Description

This project investigates service disparities within the MBTA bus network, focusing on how geography and socioeconomic demographics correlate with transit reliability. While the MBTA serves over 1 million people daily and contributes $11.5 billion in annual economic value, service is not experienced equally. Previous findings, such as the 2017 "64 Hours" report, revealed that Black bus riders spend 64 more hours per year commuting than white riders. We aim to update this narrative using post-pandemic data (2023–2024) to see if recent interventions, like fare-free pilots and bus lanes, have successfully closed these gaps.

We will analyze ridership trends, end-to-end travel times, and delay frequencies across both "Key" bus routes and high-priority lines in "transit-critical" neighborhoods like Roxbury, Dorchester, and Chelsea. By merging MBTA performance APIs with U.S. Census demographic data, this project will provide a clear visual and statistical understanding of where the system is failing its most dependent users and how service levels have shifted from pre-pandemic baselines.

---

## 2. Project Goals

Define what "success" looks like for this project. Be specific and measurable.

**Primary Goal:**  
Quantify the "reliability gap" by calculating the average delay time and wait-time variance for the 15 target bus routes (e.g., 28, 23, 111, 15) compared to system-wide averages, identifying which routes fall more than 20% below MBTA service standards.

**Secondary Goal:**  
Correlate bus performance metrics (delays, travel speed, and frequency) with neighborhood demographics (race, income, and vehicle ownership) to determine if a statistically significant disparity persists in the "time tax" paid by low-income and minority residents.

---

## 3. Data Collection Plan

Be as explicit as possible here, as requested by your instructors.

### Data Sources

**The Rider Census (2022–2024 Pooled)**  
We will utilize the MBTA’s System-Wide Passenger Survey to obtain rider demographics (race, income, vehicle ownership) at the route level.

**MBTA Open Data Portal**  
We will access the "MBTA Bus Arrival Departure Times (2018–2024)" and "Bus Ridership by Trip, Season, Route, and Stop" datasets for core performance metrics.

**MBTA V3 API**  
For real-time and granular historical reliability data, we will query the API to extract "headway" (time between buses) and "dwell time" (time spent at stops).

**Analyze Boston (City of Boston Data Portal)**  
We will download the "2020 Census for Boston" and "Neighborhood Boundaries" shapefiles to perform geospatial joins between bus routes and local demographic characteristics.

---

### Collection Method

**API Extraction**  
Use the `requests` library to programmatically fetch JSON data from the MBTA V3 API, specifically targeting the 15 priority routes identified in the LivableStreets report.

**Bulk Download**  
Use `pandas` to ingest and clean large-scale CSV files from the MassGIS Data Hub (Arrival/Departure times) and the MBTA Open Data Portal.

**Geospatial Processing**  
Utilize `GeoPandas` and the ArcGIS API for Python to map bus stop coordinates to Census Block Groups, allowing us to assign demographic traits to specific bus routes.

**Data Cleaning**  
Write functions to handle null values in actual departure times and normalize time period IDs (e.g., AM_RUSH, MIDDAY) to ensure consistency across the 2018–2024 timeframe.

---

### Data Features

**Performance**
- route_id
- direction_id
- scheduled_departure
- actual_departure
- delay_seconds
- headway_variance
- travel_time_sec

**Ridership**
- average_ons (boardings)
- average_offs (alightings)
- average_load
- season
- day_type (Weekday vs. Weekend)

**Demographics**
- %_minority_riders
- %_low_income_riders
- household_vehicle_count
- primary_trip_purpose

**Geospatial**
- stop_id
- stop_sequence
- latitude
- longitude
- neighborhood_name
- census_tract_id

---

## 4. Data Modeling

**Linear Regression (baseline model)**  
Establish interpretable relationships between demographics and delay times.

**Random Forest Regressor**  
Capture non-linear interactions between route characteristics, time of day, and neighborhood demographics.

**XGBoost**  
Handle complex feature interactions and provide feature importance rankings.

**Approach**  
Start with linear regression to identify which demographic and route features correlate most strongly with delays. Then compare performance with Random Forest and XGBoost to capture non-linear patterns.

---

## 5. Data Visualization

**Exploratory Visuals**
- Histograms
- Box plots

**Correlation Analysis**
- Heatmap of Pearson correlation coefficients

**Model Performance**
- Predicted vs. Actual scatter plots

**Dimensionality Reduction (Optional)**
- PCA or t-SNE visualization

---

## 6. Test Plan

**Validation Strategy**
- 80/20 train-test split
- 5-fold cross-validation

**Temporal Testing**
- Train on first 6 weeks
- Test on final 2 weeks

**Success Metrics**

*Regression*
- MAE
- R² Score

*Classification*
- F1 Score
- Confusion Matrix

**Baseline Comparison**
Compare against a naive baseline (predicting historical average).

---

## 7. Implementation Timeline

| Week Range | Milestone | Target Date |
|------------|-----------|------------|
| Weeks 1–2 | Data Collection & Cleaning | Feb 12, 2026 |
| Weeks 3–4 | EDA & Visualization | Feb 20, 2026 |
| Weeks 5–6 | Initial Modeling | March 6, 2026 |
| Weeks 7–8 | Model Refinement & Documentation | April 30, 2026 |
