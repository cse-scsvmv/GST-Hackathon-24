#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/train_gst.csv')
x = df.iloc[:,1:]
x.head()
print(df)



# In[2]:


print(df.isnull().sum())


# In[3]:


print(df.isnull().sum().sum())


# In[4]:


print(x.isnull().sum())


# In[5]:


columns_to_impute = ['Column6','Column8','Column15','Column0']


for column in columns_to_impute:
    if column in x.columns:  # Check if the column exists in the DataFrame
        mean_value = x[column].mean()  # Calculate the mean of the column
        x[column].fillna(mean_value, inplace=True)


# In[6]:


print(x.isnull().sum())


# In[7]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Initialize an empty DataFrame to store the final imputed values
x_final = x.copy()

# Iterate over each column to impute missing values
for column in x.columns:
    # Check if the column has missing values
    if x[column].isnull().any():
        # Prepare the data for imputation
        # Create a DataFrame for features and target
        features = x.drop(columns=column)  # Features are all columns except the one we want to impute
        target = x[column]  # Target is the column we want to impute
        
        # Split the data into training and test sets
        features_train = features[~target.isnull()]
        target_train = target[~target.isnull()]
        
        # Initialize the Decision Tree Regressor
        decision_tree = DecisionTreeRegressor(random_state=0)
        
        # Fit the model on the training data
        decision_tree.fit(features_train, target_train)
        
        # Predict the missing values
        missing_values = features[target.isnull()]
        predicted_values = decision_tree.predict(missing_values)
        
        # Fill in the missing values in the original DataFrame
        x_final.loc[x[column].isnull(), column] = predicted_values

# Display the number of missing values after imputation
print("Total missing values after imputation:", x_final.isnull().sum())


# In[8]:


import pandas as pd

# Load the dataset
file_path = 'C:/Users/shp04/OneDrive/Desktop/train_gst.csv'
df = pd.read_csv(file_path)

# Check for duplicate rows
duplicates = df[df.duplicated()]

# Display the number of duplicate rows
print("Number of duplicate rows before removing: ", duplicates.shape[0])

# Optionally display the duplicate rows
print("Duplicate rows:")
print(duplicates)

# Remove all duplicate rows, keeping the first occurrence
df_cleaned = df.drop_duplicates(keep=False)

# Display the number of duplicate rows after removal
duplicates_after = df_cleaned[df_cleaned.duplicated()]
print("Number of duplicate rows after removing: ", duplicates_after.shape[0])

# Optionally save the cleaned DataFrame to a new CSV file
cleaned_file_path = 'C:/Users/shp04/OneDrive/Desktop/balanced_trainx_cleaned.csv'
df_cleaned.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to: {cleaned_file_path}")


# # outliers
# 

# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/balanced_trainx_cleaned.csv')

# Function to identify outliers using the IQR method
def detect_outliers_iqr(data):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    
    # Calculate IQR
    IQR = Q3 - Q1
    
    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = (data < lower_bound) | (data > upper_bound)
    
    return outliers

# Columns to check for outliers
outlier_columns = ['Column0', 'Column21']

# Create a DataFrame to store the outlier information
outlier_results = pd.DataFrame()

for column in outlier_columns:
    outliers = detect_outliers_iqr(df[column])
    outlier_results[column] = outliers

# Show the results
print(outlier_results)

# Filter the original DataFrame to show only rows with outliers
outlier_data = df[outlier_results.any(axis=1)]
print("Outlier Data:")
print(outlier_data)

# --- Visualization Section ---

# Boxplot visualization for Column0 and Column21
plt.figure(figsize=(12, 6))

# 1. Boxplot for Column0
plt.subplot(1, 2, 1)
sns.boxplot(x=df['Column0'])
plt.title('Boxplot for Column0 (Outliers Highlighted)')

# 2. Boxplot for Column21
plt.subplot(1, 2, 2)
sns.boxplot(x=df['Column21'])
plt.title('Boxplot for Column21 (Outliers Highlighted)')

plt.tight_layout()
plt.show()

# Scatter plot to visualize outliers for Column0 vs Column21
plt.figure(figsize=(8, 6))
plt.scatter(df['Column0'], df['Column21'], c=outlier_results.any(axis=1), cmap='coolwarm', alpha=0.5)
plt.title('Scatter Plot of Column0 vs Column21 (Outliers Highlighted)')
plt.xlabel('Column0')
plt.ylabel('Column21')
plt.colorbar(label='Outliers')
plt.show()

# Optional: Histograms to check data distribution
plt.figure(figsize=(12, 6))

# 1. Histogram for Column0
plt.subplot(1, 2, 1)
sns.histplot(df['Column0'], bins=30, kde=True, color='blue')
plt.title('Histogram of Column0')

# 2. Histogram for Column21
plt.subplot(1, 2, 2)
sns.histplot(df['Column21'], bins=30, kde=True, color='green')
plt.title('Histogram of Column21')

plt.tight_layout()
plt.show()


# In[11]:


import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/balanced_trainx_cleaned.csv')

# Function to cap outliers using percentiles
def cap_outliers(data, column, lower_quantile=0.05, upper_quantile=0.95):
    lower_bound = data[column].quantile(lower_quantile)
    upper_bound = data[column].quantile(upper_quantile)

    # Cap values at lower and upper bound
    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
    
    return data

# Cap outliers in 'Column0' and 'Column21' to the 5th and 95th percentiles
columns_to_cap = ['Column0', 'Column21']
for column in columns_to_cap:
    df = cap_outliers(df, column)

# Display the dataset with capped outliers
print("DataFrame after capping outliers:")
print(df)


# In[12]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/train_gst.csv')

# Function to cap outliers using percentiles
def cap_outliers(data, column, lower_quantile=0.05, upper_quantile=0.95):
    lower_bound = data[column].quantile(lower_quantile)
    upper_bound = data[column].quantile(upper_quantile)

    # Cap values at lower and upper bound
    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
    
    return data

# Cap outliers in 'Column0' and 'Column21'
columns_to_cap = ['Column0', 'Column21']
for column in columns_to_cap:
    # Calculate original percentiles before capping
    original_lower_bound = df[column].quantile(0.05)
    original_upper_bound = df[column].quantile(0.95)
    
    # Cap outliers
    df = cap_outliers(df, column)
    
    # Calculate new min and max values
    new_min = df[column].min()
    new_max = df[column].max()
    
    # Output results
    print(f"Checking {column}:")
    print(f"Original 5th Percentile: {original_lower_bound}")
    print(f"Original 95th Percentile: {original_upper_bound}")
    print(f"New Min: {new_min}")
    print(f"New Max: {new_max}")
    print("-----------")

# Visualization of the data before and after capping
plt.figure(figsize=(12, 6))

# Boxplot for 'Column0'
plt.subplot(1, 2, 1)
sns.boxplot(x=df['Column0'])
plt.title('Boxplot of Column0 After Capping')

# Boxplot for 'Column21'
plt.subplot(1, 2, 2)
sns.boxplot(x=df['Column21'])
plt.title('Boxplot of Column21 After Capping')

plt.tight_layout()
plt.show()


# # PCA

# In[13]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/balanced_trainx_cleaned.csv')

# Handle missing values (choose one of the following methods)
# Option 1: Drop rows with any missing values
df_cleaned = df.dropna()

# Option 2: Fill missing values with mean
# df_cleaned = df.fillna(df.mean())

# Option 3: Forward fill
# df_cleaned = df.fillna(method='ffill')

# Option 4: Interpolation
# df_cleaned = df.interpolate()

# Select numeric features for PCA
numeric_features = df_cleaned.select_dtypes(include=['number']).values

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_features)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Plot the principal components
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.title('PCA Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()



# In[14]:


if 'target' in df.columns:
    X_train, X_test, y_train, y_test = train_test_split(pca_df[['PC1', 'PC2']], df['target'], test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))

# Optional: Anomaly Detection
reconstructed_data = pca.inverse_transform(principal_components)
reconstruction_error = np.mean((numeric_features - reconstructed_data) ** 2, axis=1)
anomaly_threshold = np.percentile(reconstruction_error, 95)  # 95th percentile as threshold
anomalies = np.where(reconstruction_error > anomaly_threshold)

print("Anomalies found at indices:", anomalies[0])  # Show indices of anomalies


# In[15]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/train_gst.csv')

# Select numeric columns for correlation
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()  # Get all numeric columns

# Calculate the correlation matrix
correlation_matrix = df[numeric_columns].corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(25, 15))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[16]:


# Visualize anomalies on the PCA plot
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], label='Normal', c='blue')
plt.scatter(pca_df.loc[anomalies[0], 'PC1'], pca_df.loc[anomalies[0], 'PC2'], label='Anomaly', c='red')
plt.title('Anomalies in PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()


# # clipping of anomalies
# 

# In[17]:


import pandas as pd
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/balanced_trainx_cleaned.csv')

# Step 2: Select columns for clipping anomalies
columns_for_clipping = ['Column0', 'Column21']
df_selected = df[columns_for_clipping].dropna().apply(pd.to_numeric, errors='coerce').dropna()

# Step 3: Calculate the IQR for clipping
Q1 = df_selected.quantile(0.25)  # 25th percentile
Q3 = df_selected.quantile(0.75)  # 75th percentile
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 4: Clip the anomalies
for col in columns_for_clipping:
    df_selected[col] = df_selected[col].clip(lower=lower_bound[col], upper=upper_bound[col])

# Step 5: Replace original columns with clipped values
df[columns_for_clipping] = df_selected

# Step 6: Save the processed dataset to a new CSV file
df.to_csv('C:/Users/shp04/OneDrive/Desktop/processed_train_gst.csv', index=False)

print("Clipping of anomalies complete. Processed data saved to 'processed_train_gst.csv'")


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_train_gst.csv')

# Step 2: Select the columns for PCA
columns_for_pca = ['Column0', 'Column21']

# Step 3: Handle missing values and ensure numeric data
df_selected = df[columns_for_pca].dropna().apply(pd.to_numeric, errors='coerce').dropna()

# Step 4: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_selected)

# Step 5: Apply PCA for dimensionality reduction
pca = PCA(n_components=1)  # Reducing to 1 principal component
X_pca = pca.fit_transform(X_scaled)

# Convert PCA result back to a DataFrame for easier manipulation
df_pca = pd.DataFrame(X_pca, columns=['PCA_Component1'])

# Step 6: Use Percentiles to identify the bounds for capping anomalies
lower_percentile = 0.01  # 1st percentile
upper_percentile = 0.99  # 99th percentile

lower_bound = df_pca['PCA_Component1'].quantile(lower_percentile)
upper_bound = df_pca['PCA_Component1'].quantile(upper_percentile)

# Step 7: Clip the values in the PCA component
df_pca['PCA_Original'] = df_pca['PCA_Component1'].copy()  # Keep the original values for comparison
df_pca['PCA_Component1'] = df_pca['PCA_Component1'].clip(lower=lower_bound, upper=upper_bound)

# Step 8: Filter the non-anomalous data
non_anomalous_data = df_pca[df_pca['PCA_Original'] == df_pca['PCA_Component1']]

# Step 9: Reset index of non-anomalous data for proper filtering
non_anomalous_data.reset_index(drop=True, inplace=True)

# Filter original dataset for non-anomalous rows using the index from non-anomalous data
df_non_anomalous_original = df_selected.iloc[non_anomalous_data.index].reset_index(drop=True)

# Step 10: Visualize the non-anomalous data
plt.figure(figsize=(10, 6))

# Histogram to show the distribution of the non-anomalous PCA values
sns.histplot(non_anomalous_data['PCA_Component1'], bins=30, kde=True, color='blue')
plt.title('Distribution of Non-Anomalous PCA Component Values')
plt.xlabel('PCA Component 1')
plt.ylabel('Frequency')
plt.grid(True)

plt.show()

# Scatter plot showing the original two columns for non-anomalous data
plt.figure(figsize=(10, 6))

# Plot non-anomalous points from the original columns
plt.scatter(df_non_anomalous_original['Column0'], 
            df_non_anomalous_original['Column21'], 
            color='green', label='Non-Anomalous Data')

plt.title('Scatter Plot of Non-Anomalous Data (Column0 vs Column21)')
plt.xlabel('Column0')
plt.ylabel('Column21')
plt.legend()
plt.grid(True)

plt.show()


# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_train_gst.csv')

# Step 2: Select columns to check for anomalies
columns_for_anomaly_check = ['Column0', 'Column21']
df_selected = df[columns_for_anomaly_check].dropna()  # Drop missing values

# Method 1: Z-Score Method
z_scores = (df_selected - df_selected.mean()) / df_selected.std()
anomalies_z = (np.abs(z_scores) > 3).any(axis=1)  # Identify anomalies
print(f"Number of anomalies detected using Z-Score: {anomalies_z.sum()}")

# Method 2: IQR Method
Q1 = df_selected.quantile(0.25)
Q3 = df_selected.quantile(0.75)
IQR = Q3 - Q1

# Define the bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify anomalies
anomalies_iqr = (df_selected < lower_bound) | (df_selected > upper_bound)
print(f"Number of anomalies detected using IQR: {anomalies_iqr.any(axis=1).sum()}")

# Visualization: Box Plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_selected)
plt.title('Box Plot to Check for Anomalies')
plt.show()

# Visualization: Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(df_selected['Column0'], df_selected['Column21'], alpha=0.5)
plt.title('Scatter Plot of Column0 vs Column21')
plt.xlabel('Column0')
plt.ylabel('Column21')
plt.grid(True)
plt.show()


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned dataset
df_cleaned = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_train_gst.csv')

# Set up the aesthetic style of the plots
sns.set(style="whitegrid")

# Plotting the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cleaned[['Column0', 'Column21']])
plt.title('Box Plot of Cleaned Data (No Anomalies)')
plt.ylabel('Values')
plt.show()

# Plotting the histogram
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Column0'], bins=30, kde=True, color='blue', label='Column0', alpha=0.5)
sns.histplot(df_cleaned['Column21'], bins=30, kde=True, color='orange', label='Column21', alpha=0.5)
plt.title('Histogram of Cleaned Data (No Anomalies)')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plotting the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['Column0'], df_cleaned['Column21'], alpha=0.5)
plt.title('Scatter Plot of Cleaned Data (No Anomalies)')
plt.xlabel('Column0')
plt.ylabel('Column21')
plt.grid()
plt.show()


# In[21]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Step 1: Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/target.csv')

# Step 2: Define your target column name
target_column_name = 'target'  # Change this to your actual target column name

# Step 3: Check class distribution
class_counts = df[target_column_name].value_counts()
print("Class Distribution:")
print(class_counts)

# Visualize class distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Class Distribution Before Balancing')
plt.xlabel('Classes')
plt.ylabel('Number of Instances')
plt.xticks(rotation=45)
plt.show()

# Step 4: Calculate the ratio of the classes
majority_class_count = class_counts.max()
minority_class_count = class_counts.min()
imbalance_ratio = majority_class_count / minority_class_count
print(f"Imbalance Ratio: {imbalance_ratio:.2f}")

# Step 5: Determine if the dataset is imbalanced based on the ratio
if imbalance_ratio > 2:  # This threshold can be adjusted based on your criteria
    print("The dataset is imbalanced.")
    
    # Step 6: Prepare features and target for SMOTE
    X = df.drop(columns=[target_column_name])  # Features
    y = df[target_column_name]                  # Target variable

    # Step 7: Convert non-numeric columns to numeric
    non_numeric_columns = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Non-numeric columns: {non_numeric_columns}")
    
    # Convert categorical variables to numeric using Label Encoding
    label_encoders = {}
    for column in non_numeric_columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le  # Save the encoder if you need to inverse transform later

    # Step 8: Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 9: Apply SMOTE to the training set
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Step 10: Check the new class distribution
    new_class_counts = pd.Series(y_res).value_counts()
    print("******----******")
    print("Class Distribution After Balancing:")
    print("the dataset is balanced")
    print(new_class_counts)

    # Visualize the new class distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=new_class_counts.index, y=new_class_counts.values)
    plt.title('Class Distribution After Balancing')
    plt.xlabel('Classes')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45)
    plt.show()

    # Step 11: Save the balanced datasets to CSV files
   # balanced_trainx_path = 'C:/Users/shp04/OneDrive/Desktop/balanced_trainx.csv'
    #balanced_trainy_path = 'C:/Users/shp04/OneDrive/Desktop/balanced_trainy.csv'

    # Convert to DataFrame and save
    #pd.DataFrame(X_res).to_csv(balanced_trainx_path, index=False)
    #pd.DataFrame(y_res, columns=[target_column_name]).to_csv(balanced_trainy_path, index=False)

    #print(f"Oversampled data saved to {balanced_trainx_path} and {balanced_trainy_path}")

#else:
 #   print("The dataset is balanced. No need for oversampling.")


# # splitting of train and test data

# In[22]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load your balanced dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_train_gst.csv')  # No spaces
target_df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/target.csv')  # No spaces

# Ensure the target variable is in a column
target_column_name = 'target'  # Adjust this to the actual column name

# Check if the target column exists
if target_column_name not in target_df.columns:
    raise ValueError(f"The target column '{target_column_name}' is not found in the target DataFrame.")

# Remove NaN values in the target variable
target_df = target_df[target_df[target_column_name].notna()]

# Reset index for both DataFrames after dropping NaN values
df.reset_index(drop=True, inplace=True)
target_df.reset_index(drop=True, inplace=True)

# Check shapes after cleaning
print(f"Shape of features DataFrame: {df.shape}")
print(f"Shape of target DataFrame: {target_df.shape}")

# Ensure the shapes match before merging
if len(df) != len(target_df):
    raise ValueError("The number of rows in the features and target variable must match after cleaning.")

# Add the target variable to the features DataFrame
df[target_column_name] = target_df[target_column_name].values  # Use values to ensure it's a NumPy array

# Display basic statistics
print("Target variable distribution before encoding:")
print(df[target_column_name].value_counts())

# Check for non-numeric values in feature columns
for column in df.columns:
    if df[column].dtype == 'object':
        print(f"Non-numeric values found in '{column}': {df[column].unique()}")

# Handle non-numeric values in feature columns
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))

# Handle NaN values in feature columns
imputer = SimpleImputer(strategy='most_frequent')
df[df.columns.difference([target_column_name])] = imputer.fit_transform(df[df.columns.difference([target_column_name])])

# Define features (X) and target (y)
X = df.drop(target_column_name, axis=1)
y = df[target_column_name]

# Split the balanced dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = decision_tree_model.predict(X_test)

# Display predictions
print("Predictions:", y_pred)
print("Unique values in predictions:", pd.Series(y_pred).value_counts())

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)  # Avoid division by zero

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[23]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Step 14: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 15: Visualize Accuracy and Precision
metrics = [accuracy, precision]
metrics_names = ['Accuracy', 'Precision']

plt.figure(figsize=(8, 4))
plt.bar(metrics_names, metrics, color=['blue', 'orange'])
plt.ylim(0, 1)  # Set the limit from 0 to 1
plt.ylabel('Score')
plt.title('Accuracy and Precision of Decision Tree Classifier')
plt.show()


# In[24]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Step 1: Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_train_gst.csv')

# Step 2: Load the target variable from a separate CSV file
target_df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/target.csv')

# Print the column names to identify the correct target column
print("Columns in target DataFrame:", target_df.columns)

# Ensure the target variable is in a column, adjust the column name if necessary
target_column_name = 'target'  # Replace with the actual name of the target column in the CSV

# Check if the number of rows in the target matches the number of rows in the features
if len(df) != len(target_df):
    raise ValueError("The number of rows in the features and target variable must match.")

# Add the target variable to the original DataFrame
df[target_column_name] = target_df[target_column_name]

# Step 3: Display the first few rows and summary of the dataset
print(df.head())
print(df.describe(include='all'))  # Get a summary of all columns

# Step 4: Check the distribution of unique values for each column
for column in df.columns:
    print(f"Unique values in '{column}': {df[column].unique()}")
    print(f"Value counts for '{column}':")
    print(df[column].value_counts(), '\n')

# Step 5: Handle NaN values in the target variable
df = df[df[target_column_name].notna()]  # Remove rows with NaN in the target variable

# Step 6: Handle non-numeric values in feature columns
# Identify categorical columns and convert them to numeric
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column is of object type (strings)
        le = LabelEncoder()  # Create a label encoder
        df[column] = le.fit_transform(df[column].astype(str))  # Convert to numeric

# Step 7: Handle NaN values in feature columns
imputer = SimpleImputer(strategy='most_frequent')  # Choose an appropriate strategy
df[df.columns.difference([target_column_name])] = imputer.fit_transform(df[df.columns.difference([target_column_name])])

# Step 8: Define features (X) and target (y)
X = df.drop(target_column_name, axis=1)  # Features
y = df[target_column_name]               # Target variable

# Step 9: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train an XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Step 11: Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Step 12: Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

# Step 13: Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Optional: Confusion Matrix and Feature Importance
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[52]:


# Step 14: Visualize Accuracy and Precision
metrics = [accuracy, precision]
metrics_names = ['Accuracy', 'Precision']

plt.figure(figsize=(8, 4))
plt.bar(metrics_names, metrics, color=['blue', 'orange'])
plt.ylim(0, 1)  # Set the limit from 0 to 1
plt.ylabel('Score')
plt.title('Accuracy and Precision of XGBoost Classifier')
plt.axhline(y=0.5, color='black', linestyle='--')  # Optional: Add a line for reference at 0.5
plt.show()


# In[26]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier

# Step 1: Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_train_gst.csv')

# Step 2: Load the target variable from a separate CSV file
target_df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/target.csv')

# Print the column names to identify the correct target column
print("Columns in target DataFrame:", target_df.columns)

# Ensure the target variable is in a column, adjust the column name if necessary
target_column_name = 'target'  # Replace with the actual name of the target column in the CSV

# Check if the number of rows in the target matches the number of rows in the features
if len(df) != len(target_df):
    raise ValueError("The number of rows in the features and target variable must match.")

# Add the target variable to the original DataFrame
df[target_column_name] = target_df[target_column_name]

# Step 3: Display the first few rows and summary of the dataset
print(df.head())
print(df.describe(include='all'))  # Get a summary of all columns

# Step 4: Handle NaN values in the target variable
df = df[df[target_column_name].notna()]  # Remove rows with NaN in the target variable

# Step 5: Handle non-numeric values in feature columns
# CatBoost can handle categorical variables directly, but we'll label encode them
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column is of object type (strings)
        le = LabelEncoder()  # Create a label encoder
        df[column] = le.fit_transform(df[column].astype(str))  # Convert to numeric

# Step 6: Handle NaN values in feature columns
imputer = SimpleImputer(strategy='most_frequent')  # Choose an appropriate strategy
df[df.columns.difference([target_column_name])] = imputer.fit_transform(df[df.columns.difference([target_column_name])])

# Step 7: Define features (X) and target (y)
X = df.drop(target_column_name, axis=1)  # Features
y = df[target_column_name]               # Target variable

# Step 8: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train a CatBoost Classifier
catboost_model = CatBoostClassifier(random_state=42, verbose=0)  # Suppressing output with verbose=0
catboost_model.fit(X_train, y_train)

# Step 10: Make predictions on the test set
y_pred = catboost_model.predict(X_test)

# Step 11: Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

# Step 12: Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 13: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - CatBoost Classifier')
plt.show()

# Step 14: Visualize Accuracy and Precision


# # cleaning the test dataset
# 

# In[29]:


import pandas as pd

# Path to the file
file_path = 'C:/Users/shp04/OneDrive/Desktop/testx.csv'

# Load the dataset
df_test = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df_test)


# In[30]:


print(df_test.isnull())


# In[31]:


print(df_test.isnull().sum())


# In[32]:


print(df_test.isnull().sum().sum())


# In[33]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset from the specified path
testx = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/testx.csv')  # Update the filename if necessary

# Step 2: Initialize a copy of the DataFrame to store the final imputed values
x_final = testx.copy()

# Step 3: Encode categorical variables if necessary
for column in x_final.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    x_final[column] = le.fit_transform(x_final[column].astype(str))  # Convert to numeric

# Step 4: Iterate over each column to impute missing values
for column in x_final.columns:
    # Check if the column has missing values
    if x_final[column].isnull().any():
        # Prepare the data for imputation
        # Create a DataFrame for features and target
        features = x_final.drop(columns=column)  # Features are all columns except the one we want to impute
        target = x_final[column]  # Target is the column we want to impute
        
        # Split the data into training sets with non-missing target values
        features_train = features[~target.isnull()]
        target_train = target[~target.isnull()]
        
        # Initialize the Decision Tree Regressor
        decision_tree = DecisionTreeRegressor(random_state=0)
        
        # Fit the model on the training data
        decision_tree.fit(features_train, target_train)
        
        # Predict the missing values
        missing_values = features[target.isnull()]
        predicted_values = decision_tree.predict(missing_values)

        # Fill in the missing values in the original DataFrame
        x_final.loc[x_final[column].isnull(), column] = predicted_values

# Step 5: Display the number of missing values after imputation
print("Total missing values after imputation:")
print(x_final.isnull().sum())


# In[34]:


import pandas as pd

# Load the dataset
file_path = 'C:/Users/shp04/OneDrive/Desktop/testx.csv'
df = pd.read_csv(file_path)

# Check for duplicate rows
duplicates = df[df.duplicated()]

# Display the number of duplicate rows
print("Number of duplicate rows before removing: ", duplicates.shape[0])

# Optionally display the duplicate rows
print("Duplicate rows:")
print(duplicates)

# Remove all duplicate rows, keeping the first occurrence
df_cleaned = df.drop_duplicates(keep=False)

# Display the number of duplicate rows after removal
duplicates_after = df_cleaned[df_cleaned.duplicated()]
print("Number of duplicate rows after removing: ", duplicates_after.shape[0])

# Optionally save the cleaned DataFrame to a new CSV file
cleaned_file_path = 'C:/Users/shp04/OneDrive/Desktop/balanced_testx_cleaned.csv'
df_cleaned.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to: {cleaned_file_path}")


# In[35]:


import pandas as pd

# Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/balanced_testx_cleaned.csv')  # Updated to your testx path

# Function to cap outliers using the IQR method
def cap_outliers_iqr(data):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    
    # Calculate IQR
    IQR = Q3 - Q1
    
    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap the outliers
    data_capped = data.clip(lower=lower_bound, upper=upper_bound)
    
    return data_capped

# Apply the outlier capping function to the relevant columns
outlier_columns = ['Column0', 'Column21']  # Add the columns you want to check

# Create a DataFrame to store the original and capped values for comparison
outlier_results = pd.DataFrame()

for column in outlier_columns:
    # Store the original data
    outlier_results[column] = df[column]
    # Cap outliers in the specified column
    df[column] = cap_outliers_iqr(df[column])

# Show the results
print("Original Outlier Data:")
print(outlier_results)

# Show the modified DataFrame after capping outliers
print("Data after capping outliers:")
print(df.head())


# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

# Create a figure with subplots
plt.figure(figsize=(14, 6))

# Plot for original data
plt.subplot(1, 2, 1)
sns.boxplot(data=outlier_results[outlier_columns])
plt.title('Original Data with Outliers')
plt.ylabel('Values')
plt.xlabel('Columns')
plt.xticks(rotation=45)

# Plot for capped data
plt.subplot(1, 2, 2)
sns.boxplot(data=df[outlier_columns])
plt.title('Capped Data')
plt.ylabel('Values')
plt.xlabel('Columns')
plt.xticks(rotation=45)

# Show the plots
plt.tight_layout()
plt.show()


# In[37]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your dataset from the specified path
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/balanced_testx_cleaned.csv')  # Updated to your testx path

# Handle missing values (choose one of the following methods)
# Option 1: Drop rows with any missing values
df_cleaned = df.dropna()

# Option 2: Fill missing values with mean
# df_cleaned = df.fillna(df.mean())

# Option 3: Forward fill
# df_cleaned = df.fillna(method='ffill')

# Option 4: Interpolation
# df_cleaned = df.interpolate()

# Select numeric features for PCA
numeric_features = df_cleaned.select_dtypes(include=['number']).values

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_features)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Plot the principal components
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.title('PCA Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()


# In[38]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset from the specified path
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/balanced_testx_cleaned.csv')  # Updated to your testx path

# Handle missing values (choose one of the following methods)
df_cleaned = df.dropna()  # You can change this as needed

# Select numeric features for PCA
numeric_features = df_cleaned.select_dtypes(include=['number']).values

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_features)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Check if 'target' exists in the DataFrame
if 'target' in df.columns:
    # Prepare data for model training
    X_train, X_test, y_train, y_test = train_test_split(pca_df[['PC1', 'PC2']], df['target'], test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))

# Anomaly Detection
reconstructed_data = pca.inverse_transform(principal_components)
reconstruction_error = np.mean((numeric_features - reconstructed_data) ** 2, axis=1)
anomaly_threshold = np.percentile(reconstruction_error, 95)  # 95th percentile as threshold
anomalies = np.where(reconstruction_error > anomaly_threshold)

if anomalies[0].size > 0:
    print("Anomalies found at indices:", anomalies[0])  # Show indices of anomalies
else:
    print("No anomalies found.")


# In[39]:


# Visualize the anomalies found in the dataset
plt.figure(figsize=(8, 6))

# Plot normal points
plt.scatter(pca_df.loc[~np.isin(pca_df.index, anomalies[0]), 'PC1'],
            pca_df.loc[~np.isin(pca_df.index, anomalies[0]), 'PC2'], 
            label='Normal', c='blue')

# Plot anomaly points
plt.scatter(pca_df.loc[anomalies[0], 'PC1'],
            pca_df.loc[anomalies[0], 'PC2'], 
            label='Anomaly', c='red')

# Titles and labels
plt.title('Anomalies in PCA Space')
plt.xlabel('PCA Component 1')  # Label for the x-axis
plt.ylabel('PCA Component 2')  # Label for the y-axis

# Legend and grid
plt.legend()
plt.grid()

# Show the plot
plt.show()


# In[40]:


import pandas as pd
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/balanced_testx_cleaned.csv')

# Step 2: Select columns for clipping anomalies
columns_for_clipping = ['Column0', 'Column21']  # Update these column names if necessary
df_selected = df[columns_for_clipping].dropna().apply(pd.to_numeric, errors='coerce').dropna()

# Step 3: Calculate the IQR for clipping
Q1 = df_selected.quantile(0.25)  # 25th percentile
Q3 = df_selected.quantile(0.75)  # 75th percentile
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 4: Clip the anomalies
for col in columns_for_clipping:
    df_selected[col] = df_selected[col].clip(lower=lower_bound[col], upper=upper_bound[col])

# Step 5: Replace original columns with clipped values
df[columns_for_clipping] = df_selected

# Step 6: Save the processed dataset to a new CSV file
df.to_csv('C:/Users/shp04/OneDrive/Desktop/processed_testx.csv', index=False)

print("Clipping of anomalies complete. Processed data saved to 'processed_testx.csv'")


# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_testx.csv')

# Step 2: Select the columns for PCA
columns_for_pca = ['Column0', 'Column21']

# Step 3: Handle missing values and ensure numeric data
df_selected = df[columns_for_pca].dropna().apply(pd.to_numeric, errors='coerce').dropna()

# Step 4: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_selected)

# Step 5: Apply PCA for dimensionality reduction
pca = PCA(n_components=1)  # Reducing to 1 principal component
X_pca = pca.fit_transform(X_scaled)

# Convert PCA result back to a DataFrame for easier manipulation
df_pca = pd.DataFrame(X_pca, columns=['PCA_Component1'])

# Step 6: Use Percentiles to identify the bounds for capping anomalies
lower_percentile = 0.01  # 1st percentile
upper_percentile = 0.99  # 99th percentile

lower_bound = df_pca['PCA_Component1'].quantile(lower_percentile)
upper_bound = df_pca['PCA_Component1'].quantile(upper_percentile)

# Step 7: Clip the values in the PCA component
df_pca['PCA_Original'] = df_pca['PCA_Component1'].copy()  # Keep the original values for comparison
df_pca['PCA_Component1'] = df_pca['PCA_Component1'].clip(lower=lower_bound, upper=upper_bound)

# Step 8: Filter the non-anomalous data
non_anomalous_data = df_pca[df_pca['PCA_Original'] == df_pca['PCA_Component1']]

# Step 9: Reset index of non-anomalous data for proper filtering
non_anomalous_data.reset_index(drop=True, inplace=True)

# Filter original dataset for non-anomalous rows using the index from non-anomalous data
df_non_anomalous_original = df_selected.iloc[non_anomalous_data.index].reset_index(drop=True)

# Step 10: Visualize the non-anomalous data
plt.figure(figsize=(10, 6))

# Histogram to show the distribution of the non-anomalous PCA values
sns.histplot(non_anomalous_data['PCA_Component1'], bins=30, kde=True, color='blue')
plt.title('Distribution of Non-Anomalous PCA Component Values')
plt.xlabel('PCA Component 1')
plt.ylabel('Frequency')
plt.grid(True)

plt.show()

# Scatter plot showing the original two columns for non-anomalous data
plt.figure(figsize=(10, 6))

# Plot non-anomalous points from the original columns
plt.scatter(df_non_anomalous_original['Column0'], 
            df_non_anomalous_original['Column21'], 
            color='green', label='Non-Anomalous Data')

plt.title('Scatter Plot of Non-Anomalous Data (Column0 vs Column21)')
plt.xlabel('Column0')
plt.ylabel('Column21')
plt.legend()
plt.grid(True)

plt.show()


# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_testx.csv')

# Step 2: Select columns to check for anomalies
columns_for_anomaly_check = ['Column0', 'Column21']
df_selected = df[columns_for_anomaly_check].dropna()  # Drop missing values

# Method 1: Z-Score Method
z_scores = (df_selected - df_selected.mean()) / df_selected.std()
anomalies_z = (np.abs(z_scores) > 3).any(axis=1)  # Identify anomalies
print(f"Number of anomalies detected using Z-Score: {anomalies_z.sum()}")

# Method 2: IQR Method
Q1 = df_selected.quantile(0.25)
Q3 = df_selected.quantile(0.75)
IQR = Q3 - Q1

# Define the bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify anomalies
anomalies_iqr = (df_selected < lower_bound) | (df_selected > upper_bound)
print(f"Number of anomalies detected using IQR: {anomalies_iqr.any(axis=1).sum()}")

# Visualization: Box Plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_selected)
plt.title('Box Plot to Check for Anomalies')
plt.show()

# Visualization: Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(df_selected['Column0'], df_selected['Column21'], alpha=0.5)
plt.title('Scatter Plot of Column0 vs Column21')
plt.xlabel('Column0')
plt.ylabel('Column21')
plt.grid(True)
plt.show()


# In[43]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned dataset
df_cleaned = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_testx.csv')

# Set up the aesthetic style of the plots
sns.set(style="whitegrid")

# Plotting the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cleaned[['Column0', 'Column21']])
plt.title('Box Plot of Cleaned Data (No Anomalies)')
plt.ylabel('Values')
plt.show()

# Plotting the histogram
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Column0'], bins=30, kde=True, color='blue', label='Column0', alpha=0.5)
sns.histplot(df_cleaned['Column21'], bins=30, kde=True, color='orange', label='Column21', alpha=0.5)
plt.title('Histogram of Cleaned Data (No Anomalies)')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plotting the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['Column0'], df_cleaned['Column21'], alpha=0.5)
plt.title('Scatter Plot of Cleaned Data (No Anomalies)')
plt.xlabel('Column0')
plt.ylabel('Column21')
plt.grid()
plt.show()


# In[46]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Step 1: Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/testy.csv')

# Step 2: Define your target column name
target_column_name = 'target'  # Change this to your actual target column name

# Step 3: Check class distribution
class_counts = df[target_column_name].value_counts()
print("Class Distribution:")
print(class_counts)

# Visualize class distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Class Distribution Before Balancing')
plt.xlabel('Classes')
plt.ylabel('Number of Instances')
plt.xticks(rotation=45)
plt.show()

# Step 4: Calculate the ratio of the classes
majority_class_count = class_counts.max()
minority_class_count = class_counts.min()
imbalance_ratio = majority_class_count / minority_class_count
print(f"Imbalance Ratio: {imbalance_ratio:.2f}")

# Step 5: Determine if the dataset is imbalanced based on the ratio
if imbalance_ratio > 2:  # This threshold can be adjusted based on your criteria
    print("The dataset is imbalanced.")
    
    # Step 6: Prepare features and target for SMOTE
    X = df.drop(columns=[target_column_name])  # Features
    y = df[target_column_name]                  # Target variable

    # Step 7: Convert non-numeric columns to numeric
    non_numeric_columns = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Non-numeric columns: {non_numeric_columns}")
    
    # Convert categorical variables to numeric using Label Encoding
    label_encoders = {}
    for column in non_numeric_columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le  # Save the encoder if you need to inverse transform later

    # Step 8: Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 9: Apply SMOTE to the training set
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Step 10: Check the new class distribution
    new_class_counts = pd.Series(y_res).value_counts()
    print("******----******")
    print("Class Distribution After Balancing:")
    print("the dataset is balanced")
    print(new_class_counts)

    # Visualize the new class distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=new_class_counts.index, y=new_class_counts.values)
    plt.title('Class Distribution After Balancing')
    plt.xlabel('Classes')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45)
    plt.show()

   
    # Convert to DataFrame and save
   # pd.DataFrame(X_res).to_csv(balanced_trainx_path, index=False)
    #pd.DataFrame(y_res, columns=[target_column_name]).to_csv(balanced_trainy_path, index=False)

   


# # using xgboost 

# In[47]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Step 1: Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_testx.csv')

# Step 2: Load the target variable from a separate CSV file
target_df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/testy.csv')

# Print the column names to identify the correct target column
print("Columns in target DataFrame:", target_df.columns)

# Ensure the target variable is in a column, adjust the column name if necessary
target_column_name = 'target'  # Replace with the actual name of the target column in the CSV

# Check if the number of rows in the target matches the number of rows in the features
if len(df) != len(target_df):
    raise ValueError("The number of rows in the features and target variable must match.")

# Add the target variable to the original DataFrame
df[target_column_name] = target_df[target_column_name]

# Step 3: Display the first few rows and summary of the dataset
print(df.head())
print(df.describe(include='all'))  # Get a summary of all columns

# Step 4: Check the distribution of unique values for each column
for column in df.columns:
    print(f"Unique values in '{column}': {df[column].unique()}")
    print(f"Value counts for '{column}':")
    print(df[column].value_counts(), '\n')

# Step 5: Handle NaN values in the target variable
df = df[df[target_column_name].notna()]  # Remove rows with NaN in the target variable

# Step 6: Handle non-numeric values in feature columns
# Identify categorical columns and convert them to numeric
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column is of object type (strings)
        le = LabelEncoder()  # Create a label encoder
        df[column] = le.fit_transform(df[column].astype(str))  # Convert to numeric

# Step 7: Handle NaN values in feature columns
imputer = SimpleImputer(strategy='most_frequent')  # Choose an appropriate strategy
df[df.columns.difference([target_column_name])] = imputer.fit_transform(df[df.columns.difference([target_column_name])])

# Step 8: Define features (X) and target (y)
X = df.drop(target_column_name, axis=1)  # Features
y = df[target_column_name]               # Target variable

# Step 9: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train an XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Step 11: Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Step 12: Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

# Step 13: Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Optional: Confusion Matrix and Feature Importance
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# # using decision tree classifier algorithm

# In[48]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_testx.csv')

# Step 2: Load the target variable from a separate CSV file
target_df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/testy.csv')

# Print the column names to identify the correct target column
print("Columns in target DataFrame:", target_df.columns)

# Ensure the target variable is in a column, adjust the column name if necessary
target_column_name = 'target'  # Replace with the actual name of the target column in the CSV

# Check if the number of rows in the target matches the number of rows in the features
if len(df) != len(target_df):
    raise ValueError("The number of rows in the features and target variable must match.")

# Add the target variable to the original DataFrame
df[target_column_name] = target_df[target_column_name]

# Step 3: Display the first few rows and summary of the dataset
print(df.head())
print(df.describe(include='all'))  # Get a summary of all columns

# Step 4: Check the distribution of unique values for each column
for column in df.columns:
    print(f"Unique values in '{column}': {df[column].unique()}")
    print(f"Value counts for '{column}':")
    print(df[column].value_counts(), '\n')

# Step 5: Handle NaN values in the target variable
df = df[df[target_column_name].notna()]  # Remove rows with NaN in the target variable

# Step 6: Handle non-numeric values in feature columns
# Identify categorical columns and convert them to numeric
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column is of object type (strings)
        le = LabelEncoder()  # Create a label encoder
        df[column] = le.fit_transform(df[column].astype(str))  # Convert to numeric

# Step 7: Handle NaN values in feature columns
imputer = SimpleImputer(strategy='most_frequent')  # Choose an appropriate strategy
df[df.columns.difference([target_column_name])] = imputer.fit_transform(df[df.columns.difference([target_column_name])])

# Step 8: Define features (X) and target (y)
X = df.drop(target_column_name, axis=1)  # Features
y = df[target_column_name]               # Target variable

# Step 9: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train a Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Step 11: Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Step 12: Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

# Step 13: Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Optional: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
# Note: For Decision Trees, feature importance can be directly obtained
importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load your dataset (features)
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_testx.csv')

# Step 2: Load the target variable from a separate CSV file
target_df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/testy.csv')

# Print the column names to identify the correct target column
print("Columns in target DataFrame:", target_df.columns)

# Ensure the target variable is in a column, adjust the column name if necessary
target_column_name = 'target'  # Replace with the actual name of the target column in the CSV

# Check if the number of rows in the target matches the number of rows in the features
if len(df) != len(target_df):
    raise ValueError("The number of rows in the features and target variable must match.")

# Add the target variable to the original DataFrame
df[target_column_name] = target_df[target_column_name]

# Step 3: Display the first few rows and summary of the dataset
print(df.head())
print(df.describe(include='all'))  # Get a summary of all columns

# Step 4: Check the distribution of unique values for each column
for column in df.columns:
    print(f"Unique values in '{column}': {df[column].unique()}")
    print(f"Value counts for '{column}':")
    print(df[column].value_counts(), '\n')

# Step 5: Handle NaN values in the target variable
df = df[df[target_column_name].notna()]  # Remove rows with NaN in the target variable

# Step 6: Handle non-numeric values in feature columns
# Identify categorical columns and convert them to numeric
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column is of object type (strings)
        le = LabelEncoder()  # Create a label encoder
        df[column] = le.fit_transform(df[column].astype(str))  # Convert to numeric

# Step 7: Handle NaN values in feature columns
imputer = SimpleImputer(strategy='most_frequent')  # Choose an appropriate strategy
df[df.columns.difference([target_column_name])] = imputer.fit_transform(df[df.columns.difference([target_column_name])])

# Step 8: Define features (X) and target (y)
X = df.drop(target_column_name, axis=1)  # Features
y = df[target_column_name]               # Target variable

# Step 9: Perform cross-validation on the dataset
# Initialize the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Perform 5-fold cross-validation and calculate multiple metrics
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(dt_model, X, y, cv=5, scoring=scoring)

# Step 10: Print cross-validation results
print(f"Cross-validation results (5-fold):")
print(f"Accuracy: {cv_results['test_accuracy']}")
print(f"Mean Accuracy: {np.mean(cv_results['test_accuracy'])}")
print(f"Precision: {cv_results['test_precision_weighted']}")
print(f"Mean Precision: {np.mean(cv_results['test_precision_weighted'])}")
print(f"Recall: {cv_results['test_recall_weighted']}")
print(f"Mean Recall: {np.mean(cv_results['test_recall_weighted'])}")
print(f"F1 Score: {cv_results['test_f1_weighted']}")
print(f"Mean F1 Score: {np.mean(cv_results['test_f1_weighted'])}")

# Step 11: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 12: Train the Decision Tree model on the training set
dt_model.fit(X_train, y_train)

# Step 13: Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Step 14: Calculate evaluation metrics on the test set
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Step 15: Print evaluation metrics
print(f"Test Set Accuracy: {accuracy}")
print(f"Test Set Precision: {precision}")
print(f"Test Set Recall: {recall}")
print(f"Test Set F1 Score: {f1}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 16: Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# # using lightgbm algorithm

# In[49]:


import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Step 1: Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_train_gst.csv')

# Step 2: Load the target variable from a separate CSV file
target_df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/target.csv')

# Print the column names to identify the correct target column
print("Columns in target DataFrame:", target_df.columns)

# Ensure the target variable is in a column, adjust the column name if necessary
target_column_name = 'target'  # Replace with the actual name of the target column in the CSV

# Check if the number of rows in the target matches the number of rows in the features
if len(df) != len(target_df):
    raise ValueError("The number of rows in the features and target variable must match.")

# Add the target variable to the original DataFrame
df[target_column_name] = target_df[target_column_name]

# Step 3: Display the first few rows and summary of the dataset
print(df.head())
print(df.describe(include='all'))  # Get a summary of all columns

# Step 4: Check the distribution of unique values for each column
for column in df.columns:
    print(f"Unique values in '{column}': {df[column].unique()}")
    print(f"Value counts for '{column}':")
    print(df[column].value_counts(), '\n')

# Step 5: Handle NaN values in the target variable
df = df[df[target_column_name].notna()]  # Remove rows with NaN in the target variable

# Step 6: Handle non-numeric values in feature columns
# Identify categorical columns and convert them to numeric
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column is of object type (strings)
        le = LabelEncoder()  # Create a label encoder
        df[column] = le.fit_transform(df[column].astype(str))  # Convert to numeric

# Step 7: Handle NaN values in feature columns
imputer = SimpleImputer(strategy='most_frequent')  # Choose an appropriate strategy
df[df.columns.difference([target_column_name])] = imputer.fit_transform(df[df.columns.difference([target_column_name])])

# Step 8: Define features (X) and target (y)
X = df.drop(target_column_name, axis=1)  # Features
y = df[target_column_name]               # Target variable

# Step 9: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train a LightGBM Classifier
lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train)

# Step 11: Make predictions on the test set
y_pred = lgbm_model.predict(X_test)

# Step 12: Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

# Step 13: Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[50]:


import matplotlib.pyplot as plt
import seaborn as sns


# Step 14: Visualize Accuracy and Precision
metrics = {'Accuracy': accuracy, 'Precision': precision}

plt.figure(figsize=(8, 5))
plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange'])
plt.ylim(0, 1)  # Set y-axis limits to 0-1 for percentage representation
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# # using catboost algorithm

# In[51]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier

# Step 1: Load your dataset
df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/processed_testx.csv')

# Step 2: Load the target variable from a separate CSV file
target_df = pd.read_csv('C:/Users/shp04/OneDrive/Desktop/testy.csv')

# Print the column names to identify the correct target column
print("Columns in target DataFrame:", target_df.columns)

# Ensure the target variable is in a column, adjust the column name if necessary
target_column_name = 'target'  # Replace with the actual name of the target column in the CSV

# Check if the number of rows in the target matches the number of rows in the features
if len(df) != len(target_df):
    raise ValueError("The number of rows in the features and target variable must match.")

# Add the target variable to the original DataFrame
df[target_column_name] = target_df[target_column_name]

# Step 3: Display the first few rows and summary of the dataset
print(df.head())
print(df.describe(include='all'))  # Get a summary of all columns

# Step 4: Check the distribution of unique values for each column
for column in df.columns:
    print(f"Unique values in '{column}': {df[column].unique()}")
    print(f"Value counts for '{column}':")
    print(df[column].value_counts(), '\n')

# Step 5: Handle NaN values in the target variable
df = df[df[target_column_name].notna()]  # Remove rows with NaN in the target variable

# Step 6: Handle non-numeric values in feature columns
# Identify categorical columns and convert them to numeric
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column is of object type (strings)
        le = LabelEncoder()  # Create a label encoder
        df[column] = le.fit_transform(df[column].astype(str))  # Convert to numeric

# Step 7: Handle NaN values in feature columns
imputer = SimpleImputer(strategy='most_frequent')  # Choose an appropriate strategy
df[df.columns.difference([target_column_name])] = imputer.fit_transform(df[df.columns.difference([target_column_name])])

# Step 8: Define features (X) and target (y)
X = df.drop(target_column_name, axis=1)  # Features
y = df[target_column_name]               # Target variable

# Step 9: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train a CatBoost Classifier
catboost_model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_seed=42, verbose=0)
catboost_model.fit(X_train, y_train)

# Step 11: Make predictions on the test set
y_pred = catboost_model.predict(X_test)

# Step 12: Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

# Step 13: Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Optional: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
# Note: CatBoost has a built-in feature importance method
importances = catboost_model.get_feature_importance()
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[ ]:




