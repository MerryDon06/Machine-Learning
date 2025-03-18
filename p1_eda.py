import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'student_sleep_patterns.csv'  
data = pd.read_csv(file_path)

# Display basic info about the dataset
print ("First 5 records:")
print(data.head())
print("---------------------------------------------------------------------")
print("\nDataset Information:")
print(data.info())
print("---------------------------------------------------------------------")
print("\nMissing Values Count:")
print(data.isnull().sum())
print("---------------------------------------------------------------------")

# Visualization of missing values
print("\nVisualizing missing values...")
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Heatmap of Missing Values")
plt.show()

# Filling missing values with median for numerical columns
data.fillna(data.median(numeric_only=True), inplace=True)

# Plotting histograms for numerical data
print("\nPlotting histograms for numerical columns...")
data.hist(figsize=(18, 12), bins=15, color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Numerical Columns", fontsize=16)
plt.show()

# Bar graph for categorical column (Gender distribution)
print("\nPlotting bar graph for categorical column (Gender)...")
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=data, palette='pastel')
plt.title("Bar Graph of Gender Distribution")
plt.show()

# Box plot to analyze Sleep Quality by University Year
print("\nPlotting box plot for Sleep Quality by University Year...")
plt.figure(figsize=(10, 6))
sns.boxplot(x='University_Year', y='Sleep_Quality', data=data, palette='muted')
plt.title("Box Plot: Sleep Quality by University Year")
plt.show()

# Scatter plot for Sleep Duration vs Study Hours
print("\nPlotting scatter plot for Sleep Duration vs Study Hours...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Study_Hours', y='Sleep_Duration', hue='Gender', data=data, palette='viridis')
plt.title("Scatter Plot: Sleep Duration vs Study Hours")
plt.show()

# Correlation heatmap for numerical columns
print("\nPlotting correlation heatmap for numerical columns...")
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()
