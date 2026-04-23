import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. Data Cleaning & Preprocessing =================
# Load the dataset
df = pd.read_csv("WorldEnergy.csv")

# Feature Selection: Extract core indicators
core_cols = ['country', 'year', 'gdp', 'population', 'primary_energy_consumption', 
             'fossil_fuel_consumption', 'renewables_consumption', 'electricity_generation']
df_clean = df[core_cols].copy()

# Time Window: Filter for data from the year 2000 onwards
df_clean = df_clean[df_clean['year'] >= 2000]

# Categorical Sampling: Extract the top 5 global economies
top5 = ['China', 'United States', 'India', 'Japan', 'Germany']
df_top5 = df_clean[df_clean['country'].isin(top5)]

# Handling Missing Values: Drop rows with NaN to ensure accurate correlation
df_top5 = df_top5.dropna()
# ====================================================================

# Set plotting style
sns.set(style="whitegrid")
plt.figure(figsize=(18, 5))

# Graph 1: Histogram - Electricity Generation Distribution
plt.subplot(1, 3, 1)
sns.histplot(df_top5['electricity_generation'], bins=20, kde=True, color='teal')
plt.title('1. Histogram: Electricity Generation Dist.')
plt.xlabel('Electricity Generation (TWh)')

# Graph 2: Boxplot - Energy Consumption Comparison by Country
plt.subplot(1, 3, 2)
sns.boxplot(data=df_top5, x='country', y='primary_energy_consumption', palette='Set2')
plt.title('2. Boxplot: Energy Consumption by Country')
plt.ylabel('Energy Consumption (TWh)')

# Graph 3: Donut Chart - Fossil Fuels vs Renewables (Cumulative since 2000)
plt.subplot(1, 3, 3)
totals = df_top5[['fossil_fuel_consumption', 'renewables_consumption']].sum()
plt.pie(totals, labels=['Fossil Fuels', 'Renewables'], autopct='%1.1f%%', 
        colors=['#ff9999','#66b3ff'], wedgeprops=dict(width=0.3))
plt.title('3. Donut Chart: Top 5 Energy Mix (Since 2000)')

plt.tight_layout()
plt.show()
plt.figure(figsize=(18, 5))

# Graph 4: Scatter Plot - GDP vs Fossil Fuel Consumption
plt.subplot(1, 3, 1)
sns.scatterplot(data=df_top5, x='gdp', y='fossil_fuel_consumption', hue='country', s=80, palette='Set1')
plt.title('4. Scatter: GDP vs Fossil Fuel Consumption')
plt.xlabel('GDP (USD)')
plt.ylabel('Fossil Fuel Cons. (TWh)')

# Graph 5: Time Series - Electricity Generation Trend (2000-Present)
plt.subplot(1, 3, 2)
sns.lineplot(data=df_top5, x='year', y='electricity_generation', hue='country', linewidth=2.5, palette='Set1')
plt.title('5. Time Series: Electricity Gen (2000-Present)')
plt.ylabel('Electricity Generation (TWh)')

# Graph 6: Bar Chart - Average Energy Consumption by Country
plt.subplot(1, 3, 3)
avg_consumption = df_top5.groupby('country')['primary_energy_consumption'].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=avg_consumption, x='country', y='primary_energy_consumption', palette='viridis')
plt.title('6. Bar Chart: Average Energy Consumption')
plt.ylabel('Average TWh')

plt.tight_layout()
plt.show()
# Graph 7: Violin Plot - GDP Distribution
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
sns.violinplot(data=df_top5, x='country', y='gdp', palette='Pastel1', inner='quartile')
plt.title('7. Violin Plot: GDP Distribution (2000-Present)')
plt.ylabel('GDP (USD)')

# Graph 8: Correlation Heatmap
plt.subplot(1, 2, 2)
corr_matrix = df_top5[['gdp', 'population', 'primary_energy_consumption', 
                       'fossil_fuel_consumption', 'renewables_consumption', 'electricity_generation']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt=".2f", linewidths=1)
plt.title('8. Correlation Heatmap (Top 5 Economies)')
plt.show()

# Graph 9: Pair Plot - Comprehensive Multivariate Matrix
pair_cols = ['country', 'gdp', 'primary_energy_consumption', 'renewables_consumption']
sns.pairplot(df_top5[pair_cols], hue='country', palette='Set1', diag_kind='kde', height=2.5)
plt.suptitle('9. Multivariate Pair Plot (Selected Variables)', y=1.02, fontsize=14)
plt.show()
