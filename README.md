# CMPSC463-Project-1-Algorithmic-Time-Series-Segmentation-and-Condition-Analysis-
## Project Overview - 
This project I have worked on will utilize multiple algorithm techniques, such as: 
divide and conquer, clustering, and kadane's algorithm, in order to analyze the data of water pumps. This data will be used to relate the findings from it towards RUL categories related to machine health. This project will not use machine learning libraries.

## Report
The full project report is available in the [`report/`](report/) folder:
- [`CMPSC463_Project1_Report.docx`](report/CMPSC463_Project1_Report.docx)

## Dataset - 
- **Source**: [Water Pump RUL - Predictive Maintenance (Kaggle)](https://www.kaggle.com/datasets/anseldsouza/water-pump-rul-predictive-maintenance)
- **Rows**: 166,441, using the first 10,000 for analysis in this project
- **Columns**: 53 (52 sensor channels, and a RUL column)
- **Sensors**: sensor_00 through sensor_51

## How to Run:
pip install -r requirements.txt (in terminal) --> run download_dataset.py
--> run main.py

## Project Structure - 
 - data/: holds the dataset once you download it through dataset_Download.py
 - src/: all source code .py files (dataLoad, task1segmentation, task2clustering, task3kadane)
 - results/: stores all the output files for each task after running main.py
   - task1_plots/
   - task2_results/
   - task3_results/
 - main.py: runs all 3 required tasks for this project
 - dataset_Download.py: will download the kaggle dataset
 - requirements.txt: all required python packages you may need to download
 - README.md: current file


## Tasks -  
 - Task 1: Segmentation
   - Uses recursion to split 10 randomly selected sensors based on their variance
   - Output will be 10 visualizations with red dashed boundary lines, and complexity scores
 - Task 2: Clustering
   - Clusters 10,000 points into 4 groups
   - Output will be cluster summaries w/majority RUL categories
 - Task 3: Kadanes Algorithm
   - Finds the max deviation intervals of all 52 sensors within the dataset
   - will output the early warning indicators, example visualization graphs, and CSV results


## Results - All outputs saved to 'results' folder:
task1_plots: 10 PNGs of segmentation
task2_results: Clustering text files
task3_results: Kadane PNGs and a CSV file