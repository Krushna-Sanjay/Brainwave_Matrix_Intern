import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Superstore_sales_dataset.csv', encoding='latin1')

print(df.head())
print(df.info())

# ---------------------Data Cleaning & Preparation-------------------------

nulls = df.isna().sum()
print(nulls)

df = df.dropna()

df = df.drop_duplicates()
print(df.info())

df = df.drop(columns=['Row ID', 'Province', 'Order Priority', 'Customer Name'])
print(df.columns)

df['Order Date'] = pd.to_datetime(df['Order Date'])


# -----------------Exploratory Data Analysis (EDA)---------------------

# Monthly Sales Trend
monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()

# Convert index from Period to Timestamp for plotting
monthly_sales.index = monthly_sales.index.to_timestamp()

# Plot using plt
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', color='blue')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(True)
plt.tight_layout()
plt.show()



# Top 10 Products by Total Sales
top_products = df.groupby(df['Product Name'])['Sales'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.title('Top 10 Products by Sales')
plt.barh(top_products.index[::-1], top_products.values[::-1], color='purple')
plt.xlabel('Products')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()



# Sales by Region
region_sales = df.groupby(df['Region'])['Sales'].sum()

plt.figure(figsize=(6,6))
plt.title('Sales by Region')
plt.pie(region_sales.values, labels=region_sales.index, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Makes the pie chart a circle')
plt.show()



# Discount vs Profit
plt.figure(figsize=(8, 5))
plt.scatter(df['Discount'], df['Profit'], alpha=0.5, color='red')
plt.title('Discount vs Profit')
plt.xlabel('Discount')
plt.ylabel('Profit')
plt.grid(True)
plt.tight_layout()
plt.show()
