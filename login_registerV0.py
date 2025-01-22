import hashlib
import random
import string
import pandas as pd
import streamlit as st
from streamlit import session_state as ss
import os

# Folder to store CSV files
csv_folder = 'data/database'

# Ensure the CSV folder exists
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)


# --- Hashing the password ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# --- Load CSV data ---
def load_csv(table_name):
    csv_path = os.path.join(csv_folder, f'{table_name}.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return pd.DataFrame()  # Return empty DataFrame if file doesn't exist


# --- Save data to CSV ---
def save_csv(df, table_name):
    csv_path = os.path.join(csv_folder, f'{table_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")


# --- Authenticate User ---
def authenticate(username, password):
    users_df = load_csv('users')
    if not users_df.empty:
        user_row = users_df[users_df['username'] == username]
        if not user_row.empty and hash_password(password) == user_row['password'].values[0]:
            return True
    return False


# --- Register User ---
def register_user(username, password):
    users_df = load_csv('users')
    # Ensure 'id' column exists, and handle it if missing
    if 'id' not in users_df.columns:
        users_df['id'] = range(1, len(users_df) + 1)  # Assign sequential IDs if missing

    # Check if the username already exists
    if username in users_df['username'].values:
        return False  # Username exists

    # Hash the password
    hashed_password = hash_password(password)

    # Find the next available ID for the new user
    new_user_id = users_df['id'].max() + 1 if not users_df.empty else 1

    # Add the new user
    new_user = pd.DataFrame({
        'id': [new_user_id],
        'username': [username],
        'password': [hashed_password]
    })

    updated_users_df = pd.concat([users_df, new_user], ignore_index=True)
    save_csv(updated_users_df, 'users')

    return True


# --- Add Farm ---
def add_farm(user_id, farm_name, location, crop_type, description):
    farms_df = load_csv('farms')
    new_farm_id = farms_df['id'].max() + 1 if not farms_df.empty else 1  # Auto-increment farm ID
    new_farm = pd.DataFrame({
        'id': [new_farm_id],
        'user_id': [user_id],
        'farm_name': [farm_name],
        'location': [location],
        'crop_type': [crop_type],
        'description': [description]
    })

    updated_farms_df = pd.concat([farms_df, new_farm], ignore_index=True)
    save_csv(updated_farms_df, 'farms')

    return new_farm_id


# --- Generate Sensor Key ---
def generate_sensor_key():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))


# --- Add Sensor ---
def add_sensor(farm_id):
    sensors_df = load_csv('sensors')
    new_sensor_key = generate_sensor_key()
    new_sensor = pd.DataFrame({
        'farm_id': [farm_id],
        'sensor_key': [new_sensor_key]
    })

    updated_sensors_df = pd.concat([sensors_df, new_sensor], ignore_index=True)
    save_csv(updated_sensors_df, 'sensors')

    return new_sensor_key


# --- Record Sensor Data ---
def record_sensor_data(sensor_key, pH, EC_level, temperature, TDS, ORP):
    sensor_data_df = load_csv('sensor_data')
    new_sensor_data_id = sensor_data_df['id'].max() + 1 if not sensor_data_df.empty else 1  # Auto-increment ID
    new_sensor_data = pd.DataFrame({
        'id': [new_sensor_data_id],
        'sensor_key': [sensor_key],
        'date': [pd.to_datetime('today')],
        'pH': [pH],
        'EC_level': [EC_level],
        'temperature': [temperature],
        'TDS': [TDS],
        'ORP': [ORP]
    })

    updated_sensor_data_df = pd.concat([sensor_data_df, new_sensor_data], ignore_index=True)
    save_csv(updated_sensor_data_df, 'sensor_data')


# --- Streamlit App Settings ---
st.set_page_config(
    page_title="AgriDT PRO",
    page_icon="🌾",
    layout="wide"
)

background_css = """
<style>
.stApp {
    background-color: #f5f5f5;
    font-family: 'Arial', sans-serif;
}
.st-emotion-cache-16txtl3 {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# Initialize session state
if "authenticated" not in ss:
    ss.authenticated = False
if "username" not in ss:
    ss.username = None
if "user_id" not in ss:
    ss.user_id = None

# Main content after login
if ss.authenticated:
    import main  # Home.py

    main.start()
    st.stop()  # Stop so that only home page is shown

# Login/Register page
st.markdown("<h1 style='text-align: center;'>Welcome to Agro Pluse Twin Hub</h1>", unsafe_allow_html=True)
st.markdown("<h4>Login or Register to Access Your Dashboard</h4>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

# Login tab
with tab1:
    st.write("#### Login")
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", placeholder="Enter your password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            ss.authenticated = True
            ss.username = username

            # Fetch user_id from 'users.csv'
            users_df = load_csv('users')
            user_row = users_df[users_df['username'] == username]
            ss.user_id =user_row.iloc[0]['id']
            st.success(f"Welcome back, {username}!")
            st.rerun()  # Reload the page
        else:
            st.error("Invalid username or password.")

# Register tab
with tab2:
    st.write("#### Register")
    new_username = st.text_input("New Username", placeholder="Choose a username")
    new_password = st.text_input("New Password", placeholder="Choose a password", type="password")
    confirm_password = st.text_input("Confirm Password", placeholder="Confirm your password", type="password")
    if st.button("Register"):
        if new_password == confirm_password and new_username:
            if register_user(new_username, new_password):
                st.success(f"User '{new_username}' registered successfully!")
                st.info("Go to the Login tab to access your dashboard.")
            else:
                st.error(f"Username '{new_username}' is already taken. Please choose another.")
        else:
            st.error("Passwords do not match or username is missing.")
