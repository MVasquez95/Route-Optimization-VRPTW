# 🚛 Route Optimization with Time Windows (VRPTW)

This project simulates a real-world logistics optimization workflow using the Solomon VRPTW benchmark dataset.  
It includes data extraction, transformation, route optimization using Google OR-Tools, and dynamic visualization of delivery routes.

## 🧩 Features
- ETL pipeline for delivery order simulation  
- Route optimization with constraints (time windows, capacity)  
- Dynamic event simulation (traffic, accidents)  
- Interactive route visualization  

## 🧱 Structure
data/
raw/ # Original Solomon datasets
processed/ # Cleaned/structured data
notebooks/
01_data_extraction_eda.ipynb
02_data_engineering_pipeline.ipynb
03_modeling_optimization.ipynb
04_demo_visualization.ipynb
src/
data_loader.py
pipeline.py
optimizer.py