# TV-Sales-Analysis-Using-EDA
📺 Exploratory Data Analysis on Television Market Data
📌 Project Overview

Online marketplaces list hundreds of television models across brands, price ranges, and technical specifications. This creates noisy, inconsistent data that makes it difficult to understand what actually drives pricing, demand, and customer perception.

This project applies web scraping and exploratory data analysis (EDA) to raw television product listings to clean the data, analyze feature patterns, and uncover key factors influencing TV prices and market positioning.

🎯 Business Problem

Retailers, analysts, and sellers struggle to:

Compare TV models fairly across brands

Understand why some models are priced significantly higher

Identify high-demand screen sizes and feature combinations

Optimize pricing and inventory decisions

Raw marketplace data is unstructured, inconsistent, and misleading without analysis.

🎯 Objectives

Scrape television product listings from an online marketplace

Clean and standardize messy product data

Analyze pricing variation across:

Brands

Screen sizes

Launch years

Sound output

Customer ratings

Identify which features actually influence price

Support data-driven inventory and pricing decisions

🧩 Dataset

Source: Online TV product listings

Format: CSV

Records: ~500 TV models

Key Columns:

Brand

Model

Screen Size (cm)

Launch Year

Sound Output (Watts)

Price

Customer Rating

📁 File: flipkart_tvs_sales.csv
(Name retained only for reference; no company mentioned in analysis.)

🛠️ Tools & Technologies

Python

Selenium

BeautifulSoup

Requests

Pandas

NumPy

Matplotlib

Seaborn

SciPy

🔄 Project Workflow

Web Scraping

Sent HTTP requests to fetch product pages

Extracted data using Selenium and BeautifulSoup

Data Cleaning & Preparation

Removed currency symbols and converted numeric fields

Extracted structured features from unstructured text

Handled missing values using median and group-based imputation

Removed duplicates and invalid records

Standardized data types across all columns

Exploratory Data Analysis

Univariate analysis

Bivariate analysis

Multivariate analysis

Correlation analysis

Hypothesis Testing

Statistical testing using Pearson correlation and t-tests

📊 Key Insights

Screen size is the strongest driver of TV pricing

Most TVs fall in the 43–55 inch range

Higher price does not guarantee higher customer ratings

Sound output has minimal impact on pricing

Most listed TVs are recent launches (2020–2024)

A small number of brands dominate the market catalog

🧪 Hypothesis Testing Summary
Hypothesis	Result
Screen size affects price	Accepted
Bigger TVs have higher sound output	Weak relationship
Average TV price is above ₹30,000	Accepted
Average screen size equals 140 cm	Rejected
📌 Business Recommendations

Focus inventory on mid-range screen sizes (43–55 inches)

Promote brands with consistent presence and strong ratings

Expand premium display technologies where margins are higher

Bundle or upsell external audio systems

Price mid-range models competitively to capture volume demand

📁 Repository Structure
├── filpkart TVs new code.py      # Web scraping + EDA code
├── flipkart_tvs_sales.csv       # Cleaned dataset
├── EDA TVs PPT new.pptx          # Presentation slides
├── README.md                    # Project documentation

🚀 How to Run the Project

Clone the repository

Install required Python libraries

Run the scraping and analysis script

Explore insights using the cleaned dataset and visualizations

👤 Author

K. Rohith Reddy
Aspiring Data Analyst | Python | SQL | Exploratory Data Analysis
