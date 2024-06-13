import streamlit as st
import pandas as pd
def delete_column(df, column_name):
    if column_name in df.columns:
        df.drop(column_name, axis=1, inplace=True)
        st.success(f"Đã xóa cột '{column_name}' thành công.")
    else:
        st.error(f"Cột '{column_name}' không tồn tại.")

def rename_column(df, old_name, new_name):
    if old_name in df.columns:
        df.rename(columns={old_name: new_name}, inplace=True)
        st.success(f"Đã đổi tên cột '{old_name}' thành '{new_name}' thành công.")
    else:
        st.error(f"Cột '{old_name}' không tồn tại.")

def convert_categorical_to_numeric(df):
    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    if len(object_columns) > 0:
        column_name = st.sidebar.selectbox("Chọn cột để chuyển đổi", object_columns)
        df[column_name] = pd.factorize(df[column_name])[0]
        st.success(f"Đã chuyển đổi biến phân loại '{column_name}' thành số thành công.")
    else:
        st.info("Không có cột dữ liệu phân loại trong tệp CSV.")

def remove_duplicates(df):
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    final_rows = df.shape[0]
    st.success(f"Đã xóa {initial_rows - final_rows} hàng trùng lặp thành công.")

def replace_null_with_mean(df, column_name):
    if column_name in df.columns:
        mean_value = df[column_name].mean()
        df[column_name].fillna(mean_value, inplace=True)
        st.success(f"Đã thay thế giá trị null trong cột '{column_name}' bằng giá trị trung bình thành công.")
    else:
        st.error(f"Cột '{column_name}' không tồn tại.")


def replace_null_with_median(df, column_name):
    if column_name in df.columns:
        median_value = df[column_name].median()
        df[column_name].fillna(median_value, inplace=True)
        st.success(f"Đã thay thế giá trị null trong cột '{column_name}' bằng giá trị trung vị thành công.")
    else:
        st.error(f"Cột '{column_name}' không tồn tại.")


def replace_null_with_custom_value(df, column_name, custom_value):
    if column_name in df.columns:
        df[column_name].fillna(custom_value, inplace=True)
        st.success(f"Đã thay thế giá trị null trong cột '{column_name}' bằng giá trị tùy chỉnh thành công.")
    else:
        st.error(f"Cột '{column_name}' không tồn tại.")


def save_file(df, filename):
    df.to_csv(filename, index=False)
    st.success(f"Đã lưu tệp '{filename}' thành công.")
