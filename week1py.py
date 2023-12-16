#!/usr/bin/env python
# coding: utf-8

# In[70]:


pip install sqlalchemy psycopg2 panda seaborn matplotlib scipy scikit-learn


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


database_name='telecom'
table_name='xdr_data'

connection_params={"host":"localhost", "user": "postgres", "password": "hi", "port":"5432", "database":database_name}

engine= create_engine(f"postgresql+psycopg2://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}")

sql_query = 'SELECT * FROM xdr_data limit 15000'

df = pd.read_sql(sql_query, con= engine)


# In[2]:


df.head(10)


# Top 10 Handsets

# In[3]:


handset_counts = df['Handset Type'].value_counts()
top_10_handsets = handset_counts.head(10)
print("Top 10 handsets used by customers:")
print(top_10_handsets)


# In[4]:


value_counts = df['Handset Type'].value_counts()[:10]
top_10_handsets = value_counts.index.tolist()
#print(value_counts)
#print(top_10_handsets)

top_10_df = pd.DataFrame(value_counts)
top_10_df.plot(kind= 'bar',rot=80)
plt.xlabel('Handset Type')
plt.ylabel('Count')
plt.title('Top 10 Handsets')
plt.show(10)


# Top Handset Manufacturer

# In[5]:


handset_manufacturer=df['Handset Manufacturer'].value_counts()
top_3_handset_manufacturers=handset_manufacturer.head(3)
print("The top handset manufacturers are: ")
print(top_3_handset_manufacturers)


# In[6]:


value_counts = df['Handset Manufacturer'].value_counts()[:3]
top_3_manufacturers = value_counts.index.tolist()

top_3_df = pd.DataFrame(value_counts)
top_3_df.plot(kind='bar', rot=45)

plt.xlabel('Handset Manufacturer')
plt.ylabel('Count')
plt.title('Top 3 Handset Manufacturers')
plt.show()


# Top Manufacturers Handset types

# In[7]:


top_3_handsetMmanufacturers=handset_manufacturer.head(3).index
filtered_df = df[df['Handset Manufacturer'].isin(top_3_handsetMmanufacturers)]
columns_of_interest=['Handset Manufacturer', 'Handset Type']
top_handsets_per_manufacturer = filtered_df.groupby(columns_of_interest).size().reset_index(name='count')
top_handsets_per_manufacturer = top_handsets_per_manufacturer.sort_values(by=['Handset Manufacturer', 'count'], ascending=[True, False])
top_5_handsets_per_manufacturer = top_handsets_per_manufacturer.groupby('Handset Manufacturer').head(5)
print("Top 5 handsets per top 3 handset manufacturers with counts:")
print(top_5_handsets_per_manufacturer)


# In[8]:


top_5_handsets = []
all_top_handsets = []
for manufacturer in top_3_manufacturers:
    manufacturer_df = df[df['Handset Manufacturer'] == manufacturer]
    handset_counts = manufacturer_df['Handset Type'].value_counts()[:5]
    all_top_handsets.append(handset_counts)
    top_5_handsets.extend(handset_counts.index.tolist())
for top_handsets in all_top_handsets:
    print(top_handsets)
    top_handsets_df = pd.DataFrame(top_handsets)
    top_handsets_df.plot(kind='bar', rot=45)
    plt.xlabel('Handset')
    plt.ylabel('Count')
    plt.title('')
    plt.show()
    
print(top_5_handsets)


# In[3]:


df['Social Media']=df['Social Media DL (Bytes)']+df['Social Media UL (Bytes)']
df['Youtube']=df['Youtube DL (Bytes)']+df['Youtube UL (Bytes)']
df['Netflix']=df['Netflix DL (Bytes)']+df['Netflix UL (Bytes)']
df['Google']=df['Google DL (Bytes)']+df['Google UL (Bytes)']
df['Email']=df['Email DL (Bytes)']+df['Email UL (Bytes)']
df['Gaming']=df['Gaming DL (Bytes)']+df['Gaming UL (Bytes)']
df['Other']=df['Other DL (Bytes)']+df['Other UL (Bytes)']
df['Total']=df['Total DL (Bytes)']+df['Total UL (Bytes)']

columns_to_show=['Bearer Id', 'Dur. (ms)','Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other', 'Total']
selected_columns_df=df[columns_to_show].tail(10)
print("Selected Columns:")
print(selected_columns_df)


# In[36]:


df.head()
aggregated_data = df.groupby("MSISDN/Number").agg({"Total": "count"})
aggregated_data.head(10)
aggregated_data.plot()
plt.ylabel("No of Xdr Sessions")
plt.xlabel("Users")
plt.show()


# In[37]:


df.head()
aggregated_data = df.groupby("MSISDN/Number").agg({"Total DL (Bytes)": "sum"})
aggregated_data['Total DL (GB)'] = aggregated_data['Total DL (Bytes)'] / 1073741824
plt.figure(figsize=(10, 6))
plt.plot(aggregated_data.index, aggregated_data['Total DL (GB)'])
plt.xlabel('MSISDN/Number')
plt.ylabel('Total Dowloaded GB')
plt.title('Total DL (GB) per MSISDN/Number')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[38]:


df.head()
aggregated_data = df.groupby("MSISDN/Number").agg({"Total UL (Bytes)": "sum"})
aggregated_data['Total UL (GB)'] = aggregated_data['Total UL (Bytes)'] / 1073741824
plt.figure(figsize=(10, 6))
plt.plot(aggregated_data.index, aggregated_data['Total UL (GB)'])
plt.xlabel('MSISDN/Number')
plt.ylabel('Total UL (GB)')
plt.title('Total UL (GB) per MSISDN/Number')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[40]:


df.head()
aggregated_data = df.groupby("MSISDN/Number").agg({"Social Media":"sum"})
aggregated_data['Social Media'] = aggregated_data['Social Media'] / 1073741824
plt.figure(figsize=(10, 6))
plt.plot(aggregated_data.index, aggregated_data['Social Media'])
plt.xlabel('MSISDN/Number')
plt.ylabel('Social Media')
plt.title('Social Media per MSISDN/Number')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[41]:


df.head()
aggregated_data = df.groupby("MSISDN/Number").agg({"Youtube":"sum"})
aggregated_data['Youtube'] = aggregated_data['Youtube'] / 1073741824
plt.figure(figsize=(10, 6))
plt.plot(aggregated_data.index, aggregated_data['Youtube'])
plt.xlabel('MSISDN/Number')
plt.ylabel('Youtube')
plt.title('Youtube per MSISDN/Number')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[42]:


df.head()
aggregated_data = df.groupby("MSISDN/Number").agg({"Netflix":"sum"})
aggregated_data['Netflix'] = aggregated_data['Netflix'] / 1073741824
plt.figure(figsize=(10, 6))
plt.plot(aggregated_data.index, aggregated_data['Netflix'])
plt.xlabel('MSISDN/Number')
plt.ylabel('Netflix')
plt.title('Netflix per MSISDN/Number')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[45]:


df.head()
aggregated_data = df.groupby("MSISDN/Number").agg({"Google":"sum"})
aggregated_data['Netflix'] = aggregated_data['Google'] / 1073741824
plt.figure(figsize=(10, 6))
plt.plot(aggregated_data.index, aggregated_data['Google'])
plt.xlabel('MSISDN/Number')
plt.ylabel('Google')
plt.title('Google per MSISDN/Number')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[46]:


df.head()
aggregated_data = df.groupby("MSISDN/Number").agg({"Email":"sum"})
aggregated_data['Email'] = aggregated_data['Email'] / 1073741824
plt.figure(figsize=(10, 6))
plt.plot(aggregated_data.index, aggregated_data['Email'])
plt.xlabel('MSISDN/Number')
plt.ylabel('Email')
plt.title('Email per MSISDN/Number')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[47]:


df.head()
aggregated_data = df.groupby("MSISDN/Number").agg({"Gaming":"sum"})
aggregated_data['Gaming'] = aggregated_data['Gaming'] / 1073741824
plt.figure(figsize=(10, 6))
plt.plot(aggregated_data.index, aggregated_data['Gaming'])
plt.xlabel('MSISDN/Number')
plt.ylabel('Gaming')
plt.title('Gaming per MSISDN/Number')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[48]:


df.head()
aggregated_data = df.groupby("MSISDN/Number").agg({"Other":"sum"})
aggregated_data['Other'] = aggregated_data['Other'] / 1073741824
plt.figure(figsize=(10, 6))
plt.plot(aggregated_data.index, aggregated_data['Other'])
plt.xlabel('MSISDN/Number')
plt.ylabel('Other')
plt.title('Other per MSISDN/Number')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[50]:


df.head()
aggregated_data = df.groupby("MSISDN/Number").agg({"Total":"sum"})
aggregated_data['Total'] = aggregated_data['Total'] / 1073741824
plt.figure(figsize=(10, 6))
plt.plot(aggregated_data.index, aggregated_data['Total'])
plt.xlabel('MSISDN/Number')
plt.ylabel('Total')
plt.title('Total per MSISDN/Number')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[52]:


from scipy.stats import zscore

missing_values = df.isnull().sum()

int_columns = df.select_dtypes(include=['int']).columns
df[int_columns] = df[int_columns].apply(lambda col: col.fillna(col.mean()))

string_columns = df.select_dtypes(include=['object']).columns
df[string_columns] = df[string_columns].fillna('Unknown')

columns_to_show=['MSISDN/Number', 'Dur. (ms)','Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other', 'Total']
selected_columns_df=df[columns_to_show].tail(10)


# Identify outliers using Z-score
z_scores = np.abs((df[int_columns] - df[int_columns].mean()) / df[int_columns].std())
z_score_threshold = 3
df_no_outliers = df[(z_scores < z_score_threshold).all(axis=1)]

# Display summary statistics before and after treatment
print("Summary Statistics Before Treatment:")
print(selected_columns_df.describe())

print("\nSummary Statistics After Treatment:")
print(df_no_outliers.describe())


# Non-Graphical Univariate Analysis by computing dispersion

# In[59]:


quantitative_variables=['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']

dispersion_parameters=pd.DataFrame({
    'Variable': quantitative_variables,
    'Range': df[quantitative_variables].max() - df[quantitative_variables].min(),
    'Variance': df[quantitative_variables].var(),
    'Standard_Deviation': df[quantitative_variables].std(),
    'Interquartile_Range': df[quantitative_variables].quantile(0.75) - df[quantitative_variables].quantile(0.25)
})

# Display the results
print("Non-Graphical Univariate Analysis - Dispersion Parameters:")
print(dispersion_parameters)


# Univariate Analysis of Social Media

# In[58]:


#df.fillna(df.mean(), inplace=True)

social_media_stats=df['Social Media']
social_media_stats=pd.to_numeric(social_media_stats, errors='coerce')
social_media_stats=social_media_stats.describe()

plt.figure(figsize=(10,6))
sns.histplot(social_media_stats, bins=20, kde=True)
plt.title('Distribution of Social Media Data')
plt.xlabel('Social Media Data (bytes)')
plt.ylabel('Frequency')
plt.show()

# Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x=social_media_stats)
plt.title('Boxplot of Social Media Data')
plt.xlabel('Social Media Data (bytes)')
plt.show()

print("Univariate Analysis -  Social Media Data:")
print(social_media_stats)


# Bivariate analysis

# In[61]:


applications = ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other', 'Total']
# Pairwise scatter plots
sns.pairplot(df[applications])
plt.suptitle('Bivariate Analysis - Relationship Between Applications and Total DL+UL', y=1.02)
plt.show()

# Correlation matrix
correlation_matrix = df[applications].corr()

# Heatmap for correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap - Applications and Total DL+UL')
plt.show()

# Display correlation coefficients
print("Correlation Coefficients:")
print(correlation_matrix)


# Segment users into the top five decile classes

# In[63]:


total_duration=df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index()
total_duration['Rank']=total_duration['Dur. (ms)'].rank(ascending=False)
total_duration['Decile']=pd.qcut(total_duration['Rank'], q=10, labels=False)
df_decile=pd.merge(df, total_duration[['MSISDN/Number', 'Decile']], on='MSISDN/Number')
total_data_per_decile=df_decile.groupby('Decile')['Total'].sum().reset_index()

# Display the result
print("Total Data (DL+UL) per Decile Class:")
print(total_data_per_decile)


# Correlation Matrix

# In[64]:


# Select relevant columns for correlation analysis
selected_columns = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

# Compute the correlation matrix
correlation_matrix = df[selected_columns].corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)


# Missing Values

# In[66]:


def percent_missing(df):

    # Calculate total number of cells in dataframe
    totalCells = np.product(df.shape)

    # Count number of missing values per column
    missingCount = df.isnull().sum()

    # Calculate total number of missing values
    totalMissing = missingCount.sum()

    # Calculate percentage of missing values
    print("The Telecom dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values.")

percent_missing(df)


# Dimensionality Reduction

# In[4]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Select relevant columns for PCA
selected_columns = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[selected_columns])

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Display the explained variance ratio for each principal component
print("Explained Variance Ratio:")
print(explained_variance_ratio)


# Task 3.1

# In[16]:


#Top 10 Customers per engagement metric:

cust_aggregation=df.groupby('MSISDN/Number').agg({
    'Dur. (ms)':'sum',
    'Total':'sum'}).reset_index()
top_10_durations=cust_aggregation.nlargest(10, 'Dur. (ms)')
print("\nTop 10 Customers by Total Session Duration:")
print (top_10_durations)

top_10_traffic = cust_aggregation.nlargest(10, 'Total')
print("\nTop 10 Customers by Total Traffic (Download and Upload):")
print(top_10_traffic)








# In[21]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df['Activity']=df['Activity Duration DL (ms)']+df['Activity Duration DL (ms)']


engagement_metrics=['Dur. (ms)', 'Activity','Total', ]
engagement_data=df[engagement_metrics]

# Standardize (normalize) the engagement data
scaler = StandardScaler()
normalized_engagement_data = scaler.fit_transform(engagement_data)

# Run k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Engagement_Cluster'] = kmeans.fit_predict(normalized_engagement_data)

# Display the resulting dataframe with engagement clusters
print("Engagement Clusters:")
print(df[['MSISDN/Number', 'Engagement_Cluster']])


# In[23]:


cluster_column = 'Engagement_Cluster'

# Select relevant columns for analysis (non-normalized metrics)
non_normalized_metrics = ['Activity', 'Dur. (ms)', 'Total']

# Group by cluster and compute metrics
cluster_metrics = df.groupby(cluster_column)[non_normalized_metrics].agg({
    'Activity': ['min', 'max', 'mean', 'sum'],
    'Dur. (ms)': ['min', 'max', 'mean', 'sum'],
    'Total': ['min', 'max', 'mean', 'sum']
}).reset_index()

# Display the results
print("Metrics for Each Cluster:")
print(cluster_metrics)


# Set the style for seaborn
sns.set(style="whitegrid")

# Plot metrics for each cluster
fig, axes = plt.subplots(nrows=len(non_normalized_metrics), ncols=4, figsize=(16, 10), sharey='row')

for i, metric in enumerate(non_normalized_metrics):
    # Plot min, max, mean, sum for each metric
    sns.barplot(x=cluster_metrics[cluster_column], y=(metric, 'min'), data=cluster_metrics, ax=axes[i, 0], palette='viridis')
    sns.barplot(x=cluster_metrics[cluster_column], y=(metric, 'max'), data=cluster_metrics, ax=axes[i, 1], palette='viridis')
    sns.barplot(x=cluster_metrics[cluster_column], y=(metric, 'mean'), data=cluster_metrics, ax=axes[i, 2], palette='viridis')
    sns.barplot(x=cluster_metrics[cluster_column], y=(metric, 'sum'), data=cluster_metrics, ax=axes[i, 3], palette='viridis')

    # Set titles for subplots
    axes[i, 0].set_title(f'Min {metric}')
    axes[i, 1].set_title(f'Max {metric}')
    axes[i, 2].set_title(f'Average {metric}')
    axes[i, 3].set_title(f'Total {metric}')

# Adjust layout
plt.tight_layout()
plt.show()


# Aggregate user total traffic per application and derive the top 10 most engaged users per application

# In[29]:


selected_columns = ['MSISDN/Number', 'Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other', 'Total']

# Aggregate total traffic per application per user
user_app_traffic = df[selected_columns].groupby('MSISDN/Number').agg({
    'Social Media': 'sum',
    'Youtube': 'sum',
    'Netflix': 'sum',
    'Google': 'sum',
    'Email': 'sum',
    'Gaming': 'sum',
    'Other': 'sum',
    'Total': 'sum'
}).reset_index()

user_app_traffic_melted = pd.melt(user_app_traffic, id_vars=['MSISDN/Number'], var_name='Application', value_name='Traffic')

# Derive the top 10 most engaged users per application
top_10_users_per_app = user_app_traffic_melted.groupby('Application').apply(lambda x: x.nlargest(10, 'Traffic')).reset_index(drop=True)

# Display the results
print("Top 10 Most Engaged Users per Application:")
print(top_10_users_per_app)


# Plot the top 3 most used applications using appropriate charts.

# In[36]:


# Sum the total traffic across all applications for each user
user_app_traffic['Total_Traffic'] = user_app_traffic.iloc[:, 1:8].sum(axis=1)

# Identify the top 3 most used applications
top_3_apps = user_app_traffic[['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']].sum().nlargest(3).index

# Filter the dataframe for the top 3 applications
top_3_apps_data = user_app_traffic[['MSISDN/Number'] + list(top_3_apps)]

# Melt the dataframe for plotting
top_3_apps_melted = pd.melt(top_3_apps_data, id_vars='MSISDN/Number', var_name='Application', value_name='Traffic')



# Plot the distribution of total traffic for the top 3 most used applications using histplot
plt.figure(figsize=(10, 6))
sns.histplot(data=top_3_apps_melted, x='Traffic', hue='Application', multiple='stack', bins=20, palette='viridis')
plt.title('Distribution of Total Traffic for Top 3 Applications')
plt.ylabel('Total Traffic')
plt.xlabel('Frequency')
plt.show()


# In[38]:


from sklearn.pipeline import make_pipeline

engagement_metrics=['Activity', 'Dur. (ms)', 'Total']
engagement_data=df[engagement_metrics]


# Standardize (normalize) the engagement data
scaler = StandardScaler()
engagement_data_scaled = scaler.fit_transform(engagement_data)

# Determine the optimal value of k using the elbow method
inertia = []
possible_k_values = range(1, 11)

for k in possible_k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    pipeline = make_pipeline(scaler, kmeans)
    pipeline.fit(engagement_data)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(possible_k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()




# Task 4- Experience Analytics

# Task 4.1

# In[44]:


from scipy.stats import zscore


df['TCP']=df['TCP DL Retrans. Vol (Bytes)']+df['TCP UL Retrans. Vol (Bytes)']
df['RTT']=df['Avg RTT DL (ms)']+df['Avg RTT UL (ms)']
df['Throughput']=df['Avg Bearer TP DL (kbps)']+df['Avg Bearer TP UL (kbps)']



# Step 1: Identify and treat missing values
int_columns=['TCP', 'RTT', 'Throughput']
df[int_columns] = df[int_columns].fillna(df[int_columns].mean())

# Replace missing values with mean

# Step 2: Identify and treat outliers using z-score
z_scores = zscore(df[int_columns])
outliers = (z_scores > 3) | (z_scores < -3)
df[int_columns] = df[int_columns].mask(outliers)

# Step 3: Replace remaining missing values with mean or mode
df[int_columns]=df[int_columns].fillna(df[int_columns].mean())  # Replace missing values with mean

# Step 4: Fill missing values for non-numeric columns
non_numeric_columns = ['Handset Type']
df[non_numeric_columns] = df[non_numeric_columns].fillna(df[non_numeric_columns].mode().iloc[0])

# Step 5: Aggregate information per customer
customer_aggregation = df.groupby('MSISDN/Number').agg({
    'TCP': 'mean',
    'RTT': 'mean',
    'Handset Type': lambda x: x.mode().iloc[0] if not x.empty else '',  # Mode for categorical variable
    'Throughput': 'mean'
}).reset_index()

# Display the results
print("Aggregated Information per Customer:")
print(customer_aggregation)


# Task 4.2 

# In[45]:


top_tcp_values = df['TCP'].value_counts().head(10)
bottom_tcp_values = df['TCP'].value_counts().tail(10)

# Most frequent TCP value
most_frequent_tcp_value = df['TCP'].mode().iloc[0]

# Compute and list the top, bottom, and most frequent values for RTT
top_rtt_values = df['RTT'].value_counts().head(10)
bottom_rtt_values = df['RTT'].value_counts().tail(10)

# Most frequent RTT value
most_frequent_rtt_value = df['RTT'].mode().iloc[0]

# Compute and list the top, bottom, and most frequent values for Throughput
top_throughput_values = df['Throughput'].value_counts().head(10)
bottom_throughput_values = df['Throughput'].value_counts().tail(10)

# Most frequent Throughput value
most_frequent_throughput_value = df['Throughput'].mode().iloc[0]

# Display the results
print("Top TCP Values:")
print(top_tcp_values)
print("\nBottom TCP Values:")
print(bottom_tcp_values)
print("\nMost Frequent TCP Value:")
print(most_frequent_tcp_value)

print("\nTop RTT Values:")
print(top_rtt_values)
print("\nBottom RTT Values:")
print(bottom_rtt_values)
print("\nMost Frequent RTT Value:")
print(most_frequent_rtt_value)

print("\nTop Throughput Values:")
print(top_throughput_values)
print("\nBottom Throughput Values:")
print(bottom_throughput_values)
print("\nMost Frequent Throughput Value:")
print(most_frequent_throughput_value)


# Task 4.3

# In[46]:


# Group by handset type and compute descriptive statistics for average throughput
throughput_stats_by_handset = df.groupby('Handset Type')['Throughput'].describe()

# Plot the distribution of average throughput per handset type
plt.figure(figsize=(12, 6))
sns.boxplot(x='Handset Type', y='Throughput', data=df, palette='viridis')
plt.title('Distribution of Average Throughput per Handset Type')
plt.xlabel('Handset Type')
plt.ylabel('Average Throughput')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()

# Display descriptive statistics
print("Descriptive Statistics for Average Throughput per Handset Type:")
print(throughput_stats_by_handset)


# In[47]:


# Group by handset type and compute the mean of TCP retransmission
tcp_retransmission_by_handset = df.groupby('Handset Type')['TCP'].mean()

# Plot the average TCP retransmission per handset type
plt.figure(figsize=(12, 6))
sns.barplot(x=tcp_retransmission_by_handset.index, y=tcp_retransmission_by_handset.values, palette='viridis')
plt.title('Average TCP Retransmission per Handset Type')
plt.xlabel('Handset Type')
plt.ylabel('Average TCP Retransmission')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()

# Display the computed average TCP retransmission per handset type
print("Average TCP Retransmission per Handset Type:")
print(tcp_retransmission_by_handset)


# Task 4.4

# In[71]:


# Select relevant columns for clustering
experience_metrics = ['TCP', 'RTT', 'Throughput']

# Subset the dataframe with the selected metrics
X = df[experience_metrics]

# Standardize (normalize) the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot clusters in 3D space (optional)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(20, 70))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=df['Cluster'], cmap='viridis', s=50)
ax.set_xlabel('TCP Retransmission')
ax.set_ylabel('RTT')
ax.set_zlabel('Average Throughput')
plt.show()

# Display cluster characteristics
cluster_characteristics = df.groupby('Cluster')[experience_metrics].mean()
print("Cluster Characteristics:")
print(cluster_characteristics)


# Task 5

# Task 5.1 

# In[79]:


from scipy.spatial import distance

# Select relevant columns for clustering
experience_metrics = ['TCP', 'RTT', 'Throughput']

# Subset the dataframe with the selected metrics
X = df[experience_metrics]

# Standardize (normalize) the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Identify the less engaged cluster (Cluster 2)
less_engaged_cluster = 2

# Calculate Euclidean distances to the less engaged cluster
df['Engagement_Score'] = df.apply(lambda row: distance.euclidean(row[experience_metrics], kmeans.cluster_centers_[less_engaged_cluster]), axis=1)

# Display the user engagement scores
print("User Engagement Scores:")
print(df[['MSISDN/Number', 'Engagement_Score']].tail(20))


# In[80]:


# Select relevant columns for clustering
experience_metrics = ['TCP', 'RTT', 'Throughput']

# Subset the dataframe with the selected metrics
X = df[experience_metrics]

# Standardize (normalize) the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Identify the cluster representing the worst experience (Cluster 2, for example)
worst_experience_cluster = 2

# Calculate Euclidean distances to the worst experience cluster
df['Experience_Score'] = df.apply(lambda row: distance.euclidean(row[experience_metrics], kmeans.cluster_centers_[worst_experience_cluster]), axis=1)

# Display the user experience scores
print("User Experience Scores:")
print(df[['MSISDN/Number', 'Experience_Score']])


# Task 5.2

# In[81]:


# Identify the less engaged cluster (Cluster 2)
less_engaged_cluster = 2

# Calculate Euclidean distances to the less engaged cluster
df['Engagement_Score'] = df.apply(lambda row: distance.euclidean(row[experience_metrics], kmeans.cluster_centers_[less_engaged_cluster]), axis=1)

# Identify the worst experience cluster (Cluster 2, for example)
worst_experience_cluster = 2

# Calculate Euclidean distances to the worst experience cluster
df['Experience_Score'] = df.apply(lambda row: distance.euclidean(row[experience_metrics], kmeans.cluster_centers_[worst_experience_cluster]), axis=1)

# Compute the satisfaction score as the average of engagement and experience scores
df['Satisfaction_Score'] = (df['Engagement_Score'] + df['Experience_Score']) / 2

# Display the top 10 satisfied customers
top_satisfied_customers = df.nlargest(10, 'Satisfaction_Score')[['MSISDN/Number', 'Satisfaction_Score']]
print("Top 10 Satisfied Customers:")
print(top_satisfied_customers)


# Task 5.3

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Task 5.4

# In[82]:


# Select relevant columns for clustering
scores = df[['Engagement_Score', 'Experience_Score']]

# Standardize (normalize) the data
scaler = StandardScaler()
scores_scaled = scaler.fit_transform(scores)

# Perform k-means clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster_Scores'] = kmeans.fit_predict(scores_scaled)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(scores_scaled[:, 0], scores_scaled[:, 1], c=df['Cluster_Scores'], cmap='viridis', s=50)
plt.title('K-means Clustering of Engagement & Experience Scores (k=2)')
plt.xlabel('Engagement Score (Standardized)')
plt.ylabel('Experience Score (Standardized)')
plt.show()

# Display the cluster assignments
print("Cluster Assignments:")
print(df[['MSISDN/Number', 'Cluster_Scores']])

