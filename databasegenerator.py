import os
import pandas as pd
import psycopg2

# PostgreSQL connection details
db_host = 'localhost'
db_port = '5432'
db_name = 'agri_dt'
db_user = 'mohsen'
db_password = 'your_new_password'

# Define the folder where CSVs will be stored
csv_folder = 'data/database'

# Make sure the folder exists
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)


# Function to get data from PostgreSQL and save it as CSV
def export_table_to_csv(table_name):
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )

    # Use pandas to load data into a DataFrame
    query = f'SELECT * FROM {table_name};'
    df = pd.read_sql(query, conn)

    # Close the connection
    conn.close()

    # Save the DataFrame to a CSV file
    csv_file_path = os.path.join(csv_folder, f'{table_name}.csv')
    df.to_csv(csv_file_path, index=False)
    print(f'Exported {table_name} to {csv_file_path}')


# List of tables to export
tables = ['farms', 'sensors', 'sensor_data', 'plants', 'plant_features', 'farm_sensors', 'feature_measurements']

# Export each table to CSV
for table in tables:
    export_table_to_csv(table)
