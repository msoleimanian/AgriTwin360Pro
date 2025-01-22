import psycopg2


# تابع اتصال به پایگاه داده
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="agri_dt",  # نام پایگاه داده
            user="mohsen",  # نام کاربری
            password="your_new_password",  # رمز عبور
            host="localhost",  # اگر PostgreSQL محلی است
            port="5432"  # پورت پیش‌فرض PostgreSQL
        )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None


# تابع برای نمایش جداول و داده‌های آن‌ها
def show_tables_and_data():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()

        # دریافت نام تمامی جداول در پایگاه داده
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
        cursor.execute(query)
        tables = cursor.fetchall()

        if tables:
            for table in tables:
                table_name = table[0]
                print(f"\nTable: {table_name}")

                # دریافت ساختار هر جدول (یعنی ستون‌ها و نوع داده‌ها)
                column_query = f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' AND table_schema = 'public';
                """
                cursor.execute(column_query)
                columns = cursor.fetchall()

                if columns:
                    print(f"Columns in table {table_name}:")
                    for column in columns:
                        print(f" - {column[0]}: {column[1]}")
                else:
                    print(f"No columns found in table {table_name}.")

                # نمایش داده‌های جدول
                data_query = f"SELECT * FROM {table_name};"
                cursor.execute(data_query)
                data = cursor.fetchall()

                if data:
                    print(f"Data in table {table_name}:")
                    for row in data:
                        print(row)
                else:
                    print(f"No data found in table {table_name}.")

        else:
            print("No tables found in the database.")

        cursor.close()
        conn.close()
    else:
        print("Error: Could not connect to the database.")


# اجرای کد
if __name__ == "__main__":
    show_tables_and_data()
