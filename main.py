import streamlit as st
from streamlit_option_menu import option_menu
import Home
import Monitoring as dt
import AI_tools as ai
import random
import string
import pandas as pd
import ExploreMenue as expmenu
import os
import Performance_Traints as pt

# Define folder to store CSV files
csv_folder = 'data/database'

# Ensure the CSV folder exists
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# --- Helper Functions ---
# Load a CSV file into a DataFrame
def load_csv(filename):
    csv_path = os.path.join(csv_folder, f'{filename}.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if the file does not exist

# Save a DataFrame to a CSV file
def save_csv(filename,df):
    csv_path = os.path.join(csv_folder, f'{filename}.csv')
    df.to_csv(csv_path, index=False)

def save_csv(df,filename):
    csv_path = os.path.join(csv_folder, f'{filename}.csv')
    df.to_csv(csv_path, index=False)

# --- Sensor Key Exists Check ---
def is_sensor_key_exists(sensor_key):
    sensors_df = load_csv('sensors')
    # Strip any whitespace and remove formatting issues
    sensors_df['sensor_key'] = sensors_df['sensor_key'].astype(str).str.strip()
    sensor_key = str(sensor_key).strip()
    return sensor_key in sensors_df['sensor_key'].values


# --- Add Farm Function ---
def add_farm(user_id, farm_name, location, crop_type, description, sensor_key, plant_features):
    st.write('sdasd')
    st.write(st.session_state)
    # Check if sensor_key exists
    if not is_sensor_key_exists(sensor_key):
        return None  # If sensor key is invalid, return None

    farms_df = load_csv('farms')
    farm_id = farms_df['id'].max() + 1 if not farms_df.empty else 1

    # Create a new farm entry
    new_farm = pd.DataFrame({
        'id': [farm_id],
        'user_id': [st.session_state.user_id],
        'farm_name': [farm_name],
        'location': [location],
        'crop_type': [crop_type],
        'description': [description]
    })

    farms_df = pd.concat([farms_df, new_farm], ignore_index=True)
    save_csv(farms_df, 'farms')

    # Add the sensor-key relationship
    farm_sensors_df = load_csv('farm_sensors')
    new_farm_sensor = pd.DataFrame({
        'farm_id': [farm_id],
        'sensor_key': [sensor_key]
    })
    farm_sensors_df = pd.concat([farm_sensors_df, new_farm_sensor], ignore_index=True)
    save_csv(farm_sensors_df, 'farm_sensors')

    # Add plant and its features
    plants_df = load_csv('plants')
    plant_id = plants_df['id'].max() + 1 if not plants_df.empty else 1
    plant_name = "Default Plant"  # You can change this to user input if required

    new_plant = pd.DataFrame({
        'id': [plant_id],
        'farm_id': [farm_id],
        'plant_name': [plant_name]
    })

    plants_df = pd.concat([plants_df, new_plant], ignore_index=True)
    save_csv(plants_df, 'plants')

    # Add plant features
    plant_features_df = load_csv('plant_features')
    for feature in plant_features:
        new_feature = pd.DataFrame({
            'plant_id': [plant_id],
            'feature_name': [feature],
            'feature_value': [None]  # Default value is None, you can customize
        })
        plant_features_df = pd.concat([plant_features_df, new_feature], ignore_index=True)

    save_csv(plant_features_df, 'plant_features')

    return farm_id

# --- Generate Sensor Key ---
def generate_sensor_key():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

# --- Start Page ---
# Main Navigation
def start():
    with st.sidebar:
        main_menu = option_menu(
            menu_title=None,
            options=["Home", "Current", "Explore", "Simulation", "View Farms", "Add Farm", "Add Plant", "Import Data"],
            icons=["house", "robot", "tools", "search", "list", "plus", "plus","plus"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#2A3B4C"},
                "icon": {"color": "white", "font-size": "20px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "5px",
                    "color": "white",
                    "--hover-color": "#6c757d"
                },
                "nav-link-selected": {
                    "background-color": "#28a745",
                    "color": "white"
                },
            }
        )

    # Handle Navigation
    if main_menu == 'Home':
        Home.createHome()  # Call your custom Home page function
    elif main_menu == 'Current':
        dt.start_monitoring()  # Call your Current monitoring function
    elif main_menu == 'Simulation':
        ai.startai()  # Call your Simulation function
    elif main_menu == 'Add Farm':
        add_farm_page()  # Call the Add Farm page function
    elif main_menu == 'Add Plant':
        add_plant_page()  # Call the Add Plant page function
    elif main_menu == 'View Farms':
        view_farms_page()  # Call your View Farms page function
    elif main_menu == 'Explore':
        pt.insightConstructor()
    elif main_menu == "Import Data":
        add_feature_measurements()

    # --- Add Farm Page ---
# Mock functions for backend operations
def is_sensor_key_exists(sensor_key):
    # Mock validation of sensor key
    return True  # Change this as needed for actual logic

def generate_sensor_key():
    # Mock function to generate a random sensor key
    return "RANDOM_SENSOR_KEY"

def add_farm(user_id, farm_name, location, description, sensor_key):
    # Mock function to save the farm to a database
    # Returns a farm_id if successful
    return 1  # Replace with actual database logic

def add_plant_to_farm(farm_id, plant_name, plant_description):
    # Mock function to save a plant to a specific farm
    return True  # Replace with actual database logic

# Mock functions for backend operations
def is_sensor_key_exists(sensor_key):
    # Mock validation of sensor key
    return True  # Change this as needed for actual logic

def generate_sensor_key():
    # Mock function to generate a random sensor key
    return "RANDOM_SENSOR_KEY"

def add_farm(user_id, farm_name, location, description, sensor_key):
    # Mock function to save the farm to a database
    # Returns a farm_id if successful
    return 1  # Replace with actual database logic

def add_plant_to_farm(farm_id, plant_name, plant_description):
    # Mock function to save a plant to a specific farm
    return True  # Replace with actual database logic

# Mock functions for backend operations
def get_all_farms(user_id):
    # Mock function to fetch all farms for the logged-in user
    # Replace with your database or API call to fetch actual farm data
    return [
        {"id": 1, "name": "Farm 1"},
        {"id": 2, "name": "Farm 2"},
        {"id": 3, "name": "Farm 3"},
    ]

def add_plant_to_farm(farm_id, plant_name, plant_features):
    # Mock function to add plant to a specific farm
    # Replace with your database or API logic
    return True

def add_farm(user_id, farm_name, location, description, sensor_key):
    # Mock function to save the farm to a database
    # Returns a farm ID if successful
    return 1  # Replace with actual database logic

# Add Farm Page
def add_farm_page():
    st.title("Create a New Farm")

    # Farm Details
    farm_name = st.text_input("Farm Name")
    location = st.text_area("Farm Location")
    description = st.text_area("Farm Description")
    sensor_key = st.text_input("Sensor Key (If empty, a random key will be generated)")

    if sensor_key:
        # Assume sensor key validation happens here
        pass
    else:
        sensor_key = "RANDOM_SENSOR_KEY"  # Generate sensor key

    if st.button("Create Farm"):
        if "user_id" in st.session_state:
            farm_id = add_farm(
                st.session_state.user_id,
                farm_name,
                location,
                description,
                sensor_key
            )
            if farm_id:
                st.success(f"Farm '{farm_name}' created successfully with sensor key '{sensor_key}'!")
            else:
                st.error("Error in creating the farm.")
        else:
            st.warning("User ID is missing. Please log in.")

# Add Plant Page
def add_plant_page():
    st.title("Add a Plant to a Farm")

    # Fetch all farms for the current user
    if "user_id" in st.session_state:
        farms = get_all_farms(st.session_state.user_id)
    else:
        st.warning("You need to log in first.")
        return

    # Dropdown to select a farm
    farm_names = [farm["name"] for farm in farms]
    selected_farm = st.selectbox("Select a Farm", farm_names)

    # Input fields for plant details
    plant_name = st.text_input("Plant Name")
    st.subheader("Plant Features (Enter one feature at a time)")
    plant_features = []

    # Add features dynamically
    while True:
        feature_name = st.text_input(f"Feature {len(plant_features) + 1} (Leave empty to finish)", key=f"feature_{len(plant_features)}")
        if feature_name:
            plant_features.append(feature_name)
        else:
            break

    # Submit button to add the plant
    if st.button("Add Plant"):
        # Find the farm ID based on the selected farm name
        farm_id = next((farm["id"] for farm in farms if farm["name"] == selected_farm), None)
        if farm_id and add_plant_to_farm(farm_id, plant_name, plant_features):
            st.success(f"Plant '{plant_name}' added successfully to '{selected_farm}' with features {plant_features}!")
        else:
            st.error("Error in adding the plant.")

# --- View Farms Page ---
def view_farms_page():
    # Load CSV files
    farms_df = load_csv('farms')
    farms_df['user_id'] = farms_df['user_id'].astype(str)
    farms_df = farms_df[farms_df['user_id'] == str(st.session_state.user_id)]
    plants_df = load_csv('plants')
    plant_features_df = load_csv('plant_features')

    st.title("Farms Overview")

    if not farms_df.empty:
        # Display the farms and features overview
        st.subheader("Farms and Features Overview")

        # Join farms with their features
        farms_with_features_df = (
            farms_df.merge(plants_df, left_on="id", right_on="farm_id", how="left")
            .merge(plant_features_df, left_on="id_y", right_on="plant_id", how="left")
        )
        farms_with_features_df = farms_with_features_df.rename(
            columns={
                "id_x": "Farm ID",
                "farm_name": "Farm Name",
                "location": "Location",
                "crop_type": "Crop Type",
                "description": "Description",
                "feature_name": "Feature Name",
                "feature_value": "Feature Value",
            }
        )
        farms_with_features_df = farms_with_features_df[
            ["Farm ID", "Farm Name", "Location", "Crop Type", "Description", "Feature Name", "Feature Value"]
        ]

        # Render as an HTML table
        st.markdown("""
            <style>
            .styled-table {
                border-collapse: collapse;
                margin: 25px 0;
                font-size: 18px;
                font-family: Arial, sans-serif;
                width: 100%;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.15);
            }
            .styled-table thead tr {
                background-color: #009879;
                color: #ffffff;
                text-align: left;
            }
            .styled-table th, .styled-table td {
                padding: 12px 15px;
                border: 1px solid #ddd;
            }
            .styled-table tbody tr {
                border-bottom: 1px solid #dddddd;
            }
            .styled-table tbody tr:nth-of-type(even) {
                background-color: #f3f3f3;
            }
            .styled-table tbody tr:last-of-type {
                border-bottom: 2px solid #009879;
            }
            </style>
        """, unsafe_allow_html=True)

        # Build HTML table
        html_table = """<table class="styled-table">
                <thead>
                    <tr>
                        <th>Farm ID</th>
                        <th>Farm Name</th>
                        <th>Location</th>
                        <th>Crop Type</th>
                        <th>Description</th>
                        <th>Feature Name</th>
                        <th>Feature Value</th>
                    </tr>
                </thead>
                <tbody>"""

        for _, row in farms_with_features_df.iterrows():
            html_table += f"""<tr>
                    <td>{row['Farm ID']}</td>
                    <td>{row['Farm Name']}</td>
                    <td>{row['Location']}</td>
                    <td>{row['Crop Type']}</td>
                    <td>{row['Description']}</td>
                    <td>{row['Feature Name']}</td>
                    <td>{row['Feature Value']}</td>
                </tr>"""

        html_table += "</tbody></table>"

        # Render the table
        st.markdown(html_table, unsafe_allow_html=True)

        # Dropdown to select a farm for detailed view
        st.subheader("Farm Details")
        farm_names = farms_df["farm_name"].tolist()
        selected_farm_name = st.selectbox("Select a farm:", farm_names)

        if selected_farm_name:
            # Fetch details of the selected farm
            selected_farm = farms_df[farms_df["farm_name"] == selected_farm_name].iloc[0]
            st.subheader(f"Details for {selected_farm_name}")
            st.write(f"**Location:** {selected_farm['location']}")
            st.write(f"**Description:** {selected_farm['description']}")

            # Fetch features for the selected farm
            farm_id = selected_farm["id"]
            selected_farm_plants = plants_df[plants_df["farm_id"] == farm_id]
            plant_ids = selected_farm_plants["id"].tolist()
            features = plant_features_df[plant_features_df["plant_id"].isin(plant_ids)]

            if not features.empty:
                for _, feature in features.iterrows():
                    st.write(f"- **Feature:** {feature['feature_name']} | **Value:** {feature['feature_value']}")
            else:
                st.warning(f"No features found for the selected farm '{selected_farm_name}'.")
    else:
        st.warning("No farms available.")

def add_feature_measurements():
    selected_farm = st.sidebar.selectbox("Select the Farm:", ["UCTC", "INTROP"])
    selected_farm = st.sidebar.selectbox("Select the Plant:", ["Pak Choy-Side1", "Pak Choy-Side2"])
    # Load CSV files
    farms_df = load_csv('farms')
    farms_df['user_id'] = farms_df['user_id'].astype(str)
    farms_df = farms_df[farms_df['user_id'] == str(st.session_state.user_id)]
    plants_df = load_csv('plants')
    plant_features_df = load_csv('plant_features')
    feature_measurements_df = load_csv('feature_measurements')

    st.title("Add New Feature Measurements")

    if not farms_df.empty:
        # Dropdown to select a farm
        farm_names = farms_df["farm_name"].tolist()
        selected_farm_name = farm_names[0]

        if selected_farm_name:
            # Fetch details of the selected farm
            selected_farm = farms_df[farms_df["farm_name"] == selected_farm_name].iloc[0]
            farm_id = selected_farm["id"]

            # Fetch features for the selected farm
            selected_farm_plants = plants_df[plants_df["farm_id"] == farm_id]
            plant_ids = selected_farm_plants["id"].tolist()
            features = plant_features_df[plant_features_df["plant_id"].isin(plant_ids)]

            st.subheader(f"Add Measurement for {selected_farm_name}")
            if not features.empty:
                selected_feature_name = st.selectbox("Select a feature:", features["feature_name"].unique())
                new_value = st.text_input("Enter the new value for the feature")
                measurement_date = st.date_input("Measurement Date", value=pd.Timestamp.now().date())

                if st.button("Add Measurement"):
                    if selected_feature_name and new_value:
                        # Append to feature_measurements_df
                        new_row = {
                            "plant_id": plant_ids[0],  # Assuming the first plant ID
                            "feature_name": selected_feature_name,
                            "feature_value": new_value,
                            "measurement_date": measurement_date,
                        }
                        feature_measurements_df = pd.concat([feature_measurements_df, pd.DataFrame([new_row])],
                                                            ignore_index=True)
                        save_csv(feature_measurements_df, 'feature_measurements')  # Save back to the CSV
                        st.success(f"Measurement for feature '{selected_feature_name}' added successfully!")
                    else:
                        st.warning("Please select a feature and enter a value.")
            else:
                st.warning(f"No features found for the selected farm '{selected_farm_name}'.")
    else:
        st.warning("No farms available.")

    with st.expander("Upload Image and Set Date (Optional)"):
        uploaded_image = st.file_uploader("Upload an image (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
        upload_date = st.date_input("Select a date for the image", value=pd.Timestamp.now().date())

        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image Preview", use_column_width=True)
            st.write(f"Selected Date: {upload_date}")


