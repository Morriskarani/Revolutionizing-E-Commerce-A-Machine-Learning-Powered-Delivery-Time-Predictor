import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv('E_Commerce.csv')

# Displaying the first few rows of the dataframe
print(df.head())

# Checking the shape of the dataset
print(f"Shape of the dataset: {df.shape}")

# Checking data types of the columns
print(f"Data types:\n{df.dtypes}")

# Statistical summary of the dataset
print(df.describe())

# Check if 'Gender' column exists
if 'Gender' in df.columns:
    gender_counts = df['Gender'].value_counts()
    
    # Check if there are values in the 'Gender' column
    if not gender_counts.empty:
        # Gender Distribution Pie Chart
        plt.figure(figsize=(6, 6))
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Gender Distribution')
        plt.show()
    else:
        print("The 'Gender' column is empty or has no values.")
else:
    print("The 'Gender' column does not exist in the dataset.")
# Creating subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Check if the columns exist before plotting
required_columns = ['Weight_in_gms', 'Product_importance', 'Cost_of_the_Product']

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
else:
    # Creating subplots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Weight Distribution with KDE
    sns.histplot(df['Weight_in_gms'], ax=ax[0], kde=True).set_title('Weight Distribution')

    # Countplot for Product Importance
    sns.countplot(x='Product_importance', data=df, ax=ax[1]).set_title('Product Importance')

    # Cost of the Product with KDE
    sns.histplot(df['Cost_of_the_Product'], ax=ax[2], kde=True).set_title('Cost of the Product')

    # Adjust layout for better readability
    plt.tight_layout()

    # Display the plots
    plt.show()


