import psycopg2
import hashlib
import random
import string
import streamlit as st
from streamlit import session_state as ss


# --- اتصال به پایگاه داده PostgreSQL ---
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="agri_dt",  # نام پایگاه داده
            user="mohsen",      # نام کاربری
            password="your_new_password",  # رمز عبور جدید شما
            host="localhost",   # اگر PostgreSQL محلی است
            port="5432"         # پورت پیش‌فرض PostgreSQL
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None

# --- هش کردن رمز عبور ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- اعتبارسنجی ورود کاربر ---
def authenticate(username, password):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result and hash_password(password) == result[0]:
            return True
        return False
    return False

# --- ثبت‌نام کاربر ---
def register_user(username, password):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        hashed_password = hash_password(password)
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except psycopg2.errors.UniqueViolation:
            conn.rollback()  # در صورت تکراری بودن نام کاربری
            cursor.close()
            conn.close()
            return False
    return False

# --- اضافه کردن مزرعه ---
def add_farm(user_id, farm_name, location, crop_type, description):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO farms (user_id, farm_name, location, crop_type, description) VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (user_id, farm_name, location, crop_type, description)
        )
        farm_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        return farm_id
    return None

# --- تولید کد سنسور منحصر به فرد ---
def generate_sensor_key():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))  # کد سنسور 10 کاراکتری

# --- اضافه کردن سنسور ---
def add_sensor(farm_id):
    sensor_key = generate_sensor_key()  # تولید کد سنسور
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sensors (farm_id, sensor_key) VALUES (%s, %s) RETURNING id",
            (farm_id, sensor_key)
        )
        sensor_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        return sensor_key
    return None

# --- ثبت داده‌های سنسور ---
def record_sensor_data(sensor_id, pH, EC_level, temperature, TDS, ORP):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO environmental_factors (sensor_id, date, pH, EC_level, temperature, TDS, ORP) VALUES (%s, CURRENT_DATE, %s, %s, %s, %s, %s)",
            (sensor_id, pH, EC_level, temperature, TDS, ORP)
        )
        conn.commit()
        cursor.close()
        conn.close()


# --- تنظیمات Streamlit ---
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

# --- آغاز کاربر ---
if "authenticated" not in ss:
    ss.authenticated = False
if "username" not in ss:
    ss.username = None
if "user_id" not in ss:
    ss.user_id = None

# --- صفحه اصلی پس از ورود ---
if ss.authenticated:
    import main  # Home.py

    main.start()
    st.stop()  # توقف تا فقط صفحه اصلی نمایش داده شود

# --- صفحه ورود/ثبت‌نام ---
st.markdown("<h1 style='text-align: center;'>Welcome to Agro Pluse Twin Hub</h1>", unsafe_allow_html=True)
st.markdown("<h4>Login or Register to Access Your Dashboard</h4>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

# --- صفحه ورود ---
with tab1:
    st.write("#### Login")
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", placeholder="Enter your password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            # اعتبارسنجی موفق
            ss.authenticated = True
            ss.username = username
            # یافتن شناسه کاربری از پایگاه داده
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            ss.user_id = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            st.success(f"Welcome back, {username}!")
            st.rerun()  # برای بارگذاری دوباره صفحه
        else:
            st.error("Invalid username or password.")

# --- صفحه ثبت‌نام ---
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