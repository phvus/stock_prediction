from vnstock import Listing
import pandas as pd
import os
import dotenv
import psycopg2
from psycopg2 import OperationalError
dotenv.load_dotenv()
List = []
with open('VN100.txt', 'r') as file:
    List = file.read().splitlines()
for symbol in List:
    print(symbol)
# def get_conn():
#     try:
#         conn = psycopg2.connect(
#             host=os.getenv("DB_HOST"),
#             port=os.getenv("DB_PORT"),
#             dbname=os.getenv("DB_NAME"),
#             user=os.getenv("DB_USER"),
#             password=os.getenv("DB_PASSWORD")
#         )
#         return conn
#     except OperationalError as e:
#         # log error (ở đây in ra console)
#         print("Không thể kết nối tới PostgreSQL:", e)
#         return None

# Create_table = """
# CREATE TABLE IF NOT EXISTS company_info(
#     symbol VARCHAR(255) PRIMARY KEY,
#     icb_name2 VARCHAR(255),
#     listing_date DATE,
#     ceo_name VARCHAR(255)
# );"""
# conn = get_conn()


# cur = conn.cursor()
# cur.execute(Create_table)
# conn.commit()
# cur.close()
# conn.close()
# print("Đã tạo bảng company_info thành công!")