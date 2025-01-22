import psycopg2
import random
from datetime import datetime, timedelta


# Connect to the database
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="agri_dt",  # Database name
            user="mohsen",  # Username
            password="your_new_password",  # Password
            host="localhost",  # Host (localhost for local setup)
            port="5432"  # Default PostgreSQL port
        )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None


# Drop existing tables and create new ones
def drop_and_create_tables():
    drop_farms_table = "DROP TABLE IF EXISTS farms CASCADE;"
    drop_sensors_table = "DROP TABLE IF EXISTS sensors CASCADE;"
    drop_sensor_data_table = "DROP TABLE IF EXISTS sensor_data CASCADE;"
    drop_plants_table = "DROP TABLE IF EXISTS plants CASCADE;"
    drop_plant_features_table = "DROP TABLE IF EXISTS plant_features CASCADE;"
    drop_farm_sensors_table = "DROP TABLE IF EXISTS farm_sensors CASCADE;"
    drop_feature_measurements_table = "DROP TABLE IF EXISTS feature_measurements CASCADE;"

    # Add feature_measurements table to track feature changes over time
    create_feature_measurements_table = """
    CREATE TABLE IF NOT EXISTS feature_measurements (
        id SERIAL PRIMARY KEY,
        plant_id INT REFERENCES plants(id) ON DELETE CASCADE,
        feature_name VARCHAR(255) NOT NULL,
        feature_value DECIMAL NOT NULL,
        measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Table creation queries
    create_sensors_table = """
    CREATE TABLE IF NOT EXISTS sensors (
        sensor_key VARCHAR(255) PRIMARY KEY,
        sensor_type VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    create_farms_table = """
    CREATE TABLE IF NOT EXISTS farms (
        id SERIAL PRIMARY KEY,
        user_id INT NOT NULL,
        farm_name VARCHAR(255) NOT NULL,
        location TEXT,
        crop_type VARCHAR(100),
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    create_sensor_data_table = """
    CREATE TABLE IF NOT EXISTS sensor_data (
        id SERIAL PRIMARY KEY,
        sensor_key VARCHAR(255) REFERENCES sensors(sensor_key) ON DELETE CASCADE,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ph_level DECIMAL,
        ec_level DECIMAL,
        temperature DECIMAL,
        tds DECIMAL,
        orp DECIMAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    create_plants_table = """
    CREATE TABLE IF NOT EXISTS plants (
        id SERIAL PRIMARY KEY,
        farm_id INT REFERENCES farms(id) ON DELETE CASCADE,
        plant_name VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    create_plant_features_table = """
    CREATE TABLE IF NOT EXISTS plant_features (
        id SERIAL PRIMARY KEY,
        plant_id INT REFERENCES plants(id) ON DELETE CASCADE,
        feature_name VARCHAR(255) NOT NULL,
        feature_value DECIMAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    create_farm_sensors_table = """
    CREATE TABLE IF NOT EXISTS farm_sensors (
        farm_id INT REFERENCES farms(id) ON DELETE CASCADE,
        sensor_key VARCHAR(255) REFERENCES sensors(sensor_key) ON DELETE CASCADE,
        PRIMARY KEY (farm_id, sensor_key)
    );
    """

    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        # Drop tables
        cursor.execute(drop_farms_table)
        cursor.execute(drop_sensors_table)
        cursor.execute(drop_sensor_data_table)
        cursor.execute(drop_plants_table)
        cursor.execute(drop_plant_features_table)
        cursor.execute(drop_farm_sensors_table)
        cursor.execute(drop_feature_measurements_table)

        # Create tables
        cursor.execute(create_sensors_table)
        cursor.execute(create_farms_table)
        cursor.execute(create_sensor_data_table)
        cursor.execute(create_plants_table)
        cursor.execute(create_plant_features_table)
        cursor.execute(create_farm_sensors_table)
        cursor.execute(create_feature_measurements_table)

        conn.commit()
        cursor.close()
        conn.close()
        print("Tables created successfully.")
    else:
        print("Error: Could not connect to the database.")


# Check if a sensor key exists
def check_sensor_key(sensor_key):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 1 FROM sensors WHERE sensor_key = %s
        """, (sensor_key,))
        exists = cursor.fetchone() is not None

        cursor.close()
        conn.close()
        return exists
    else:
        print("Error: Could not connect to the database.")
        return False


# Create a sensor key if it doesn't exist
def create_sensor_key(sensor_key):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO sensors (sensor_key, sensor_type)
            VALUES (%s, 'Default')
            ON CONFLICT (sensor_key) DO NOTHING
        """, (sensor_key,))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"Sensor key '{sensor_key}' created successfully.")
    else:
        print("Error: Could not connect to the database.")


# Insert data into feature_measurements
def insert_feature_measurement(plant_id, feature_name, feature_value):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO feature_measurements (plant_id, feature_name, feature_value, measurement_date)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        """, (plant_id, feature_name, feature_value))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"Measurement for {feature_name} added successfully.")
    else:
        print("Error: Could not connect to the database.")


# Insert fake data for testing
def insert_fake_data(sensor_key):
    # Check if the sensor key exists, create it if not
    if not check_sensor_key(sensor_key):
        print(f"Sensor key '{sensor_key}' does not exist. Creating it.")
        create_sensor_key(sensor_key)

    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()

        # Insert a farm
        cursor.execute("""
            INSERT INTO farms (user_id, farm_name, location, crop_type, description)
            VALUES (1, 'Farm 1', 'Test Location', 'Wheat', 'Test Description') RETURNING id
        """)
        farm_id = cursor.fetchone()[0]

        # Insert a plant
        cursor.execute("""
            INSERT INTO plants (farm_id, plant_name)
            VALUES (%s, 'Wheat Plant') RETURNING id
        """, (farm_id,))
        plant_id = cursor.fetchone()[0]

        conn.commit()

        # Insert a feature
        cursor.execute("""
            INSERT INTO plant_features (plant_id, feature_name, feature_value)
            VALUES (%s, 'Plant Height', 50.5)
        """, (plant_id,))
        conn.commit()

        # Add historical measurements for the feature
        start_date = datetime.now() - timedelta(days=15)
        for day in range(15):
            measurement_date = start_date + timedelta(days=day)
            feature_value = random.uniform(40.0, 100.0)  # Random height value
            insert_feature_measurement(plant_id, 'Plant Height', feature_value)

        cursor.close()
        conn.close()
        print("Fake data inserted successfully.")
    else:
        print("Error: Could not connect to the database.")


# Execute the code
if __name__ == "__main__":
    # Drop existing tables and recreate them
    drop_and_create_tables()

    # Insert fake data with a sensor key
    insert_fake_data("1001")
