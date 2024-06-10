import streamlit as st
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, ttk
from tkinter import *
from tkinter.ttk import *
import time
import _thread
import numpy as np
import seaborn as sns
import random
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier  
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, mean_squared_error


def display_data_info(df):
    st.subheader("Thông tin dữ liệu:")
    st.write("Số lượng hàng:", df.shape[0])
    st.write("Số lượng cột:", df.shape[1])
    st.write("Số lượng giá trị không null:")
    st.write(df.notnull().sum())
    st.write("Số lượng giá trị null:")
    st.write(df.isnull().sum())
    st.write("Kiểu dữ liệu:")
    st.write(df.dtypes)
    st.write("Mô tả dữ liệu:")
    st.write(df.describe())

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

def main():
    
    st.set_page_config(page_title="Data Prediction", page_icon="🖥")
    st.title("📑 :blue[Thông tin dữ liệu]")
        
    logo = Image.open("img/logo.jpg")
    st.markdown("##")
    #side bar
    st.sidebar.image("img/logo.jpg")

    uploaded_file = st.sidebar.file_uploader("📁 Tải lên tệp CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dữ liệu ban đầu")
        st.dataframe(df)

        st.sidebar.title("🛠 Tiền xử lí dữ liệu")

        if st.sidebar.checkbox("Xóa cột"):
            column_to_delete = st.sidebar.selectbox("Chọn cột để xóa", df.columns)
            delete_column(df, column_to_delete)

        if st.sidebar.checkbox("Đổi tên cột"):
            old_column_name = st.sidebar.selectbox("Chọn cột cần đổi tên", df.columns)
            new_column_name = st.sidebar.text_input("Nhập tên mới cho cột", value=old_column_name)
            rename_column(df, old_column_name, new_column_name)

        if st.sidebar.checkbox("Chuyển đổi biến phân loại thành số"):
            convert_categorical_to_numeric(df)

        if st.sidebar.button("Xóa hàng trùng lặp"):
            remove_duplicates(df)

        if st.sidebar.checkbox("Thay thế giá trị null"):
            column_to_replace = st.sidebar.selectbox("Chọn cột để thay thế giá trị null", df.columns)
            replace_method = st.sidebar.selectbox("Phương pháp thay thế", ["Xóa", "Trung bình", "Trung vị", "Tùy chỉnh"])

            if replace_method == "Xóa":
                df.dropna(subset=[column_to_replace], inplace=True)
                st.success(f"Đã xóa hàng chứa giá trị null trong cột '{column_to_replace}' thành công.")
            elif replace_method == "Trung bình":
                replace_null_with_mean(df, column_to_replace)
            elif replace_method == "Trung vị":
                replace_null_with_median(df, column_to_replace)
            elif replace_method == "Tùy chỉnh":
                custom_value = st.sidebar.text_input("Nhập giá trị tùy chỉnh", value="")
                replace_null_with_custom_value(df, column_to_replace, custom_value)

        if st.sidebar.button("⬇ Lưu tệp"):
            save_file(df, "edited_file.csv")

        st.subheader("Dữ liệu sau chỉnh sửa")
        st.dataframe(df)
        display_data_info(df)

        st.title("📊 :green[Thực hiện Train mô hình]")

        col1, spacer, col2 = st.columns([2, 1, 2])

        with col1:
            multiselect_x = st.multiselect(
                "Biến độc lập x:",
                options=df.columns
            )
        
        with col2:
            select_y = st.selectbox(
                "Biến phụ thuộc y:",
                df.columns
            )

        select_model = st.selectbox(
            "Chọn mô hình: ",
            ["Chọn mô hình thực hiện","Hồi quy tuyến tính", "Hồi quy Logistic", "Hồi quy KNN", "DecisionTreeClassifier", "DecisionTreeRegressor", "Random Forest"]
        )

        if st.button("Chạy mô hình"):

            # Tạo các mảng numpy cho biến đầu vào và biến mục tiêu
            input_vars = multiselect_x
            count_input = len(input_vars)
            target_var = select_y

            X = df[input_vars]
            y = df[target_var]
            # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if select_model == "Hồi quy tuyến tính":    
                model = LinearRegression().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = model.score(X_test, y_test)

                st.markdown("<h3 style='text-align: center;color: red'>Biểu đồ hồi quy tuyến tính</h3>", unsafe_allow_html=True)
                # st.header(":red[Biểu đồ hồi quy tuyến tính]")
                if count_input == 1:
                    plt.scatter(X_test, y_test, color ='b',label='Actual')
                    plt.plot(X_test, y_pred, color ='r',label='Predicted')
                    plt.show()
                elif count_input > 1:
                    X_test_values = X_test.values
                    for i in range(X_test.shape[1]):
                        plt.figure(figsize=(12, 6))
                        plt.scatter(X_test_values[:, i], y_test, color ='b', label='Actual')
                        plt.scatter(X_test_values[:, i], y_pred, color ='r', label='Predicted')
                        plt.xlabel('Feature {}'.format(i))
                        plt.ylabel('Output')
                        plt.legend()
                        plt.show()

                st.pyplot(plt)
                # st.caption(f":green[Độ chính xác mô hình là: {accuracy}]")
                st.markdown(f"<h6 style='text-align: center;color: green'>Độ chính xác mô hình là:{accuracy}</h6>", unsafe_allow_html=True)

            elif select_model == "Hồi quy Logistic":
                # st.header(":red[Biểu đồ hồi quy Logistic]")
                st.markdown("<h3 style='text-align: center;color: red'>Biểu đồ hồi quy Logistic</h3>", unsafe_allow_html=True)

                cols = X_train.columns
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                X_train = pd.DataFrame(X_train, columns=[cols])
                X_test = pd.DataFrame(X_test, columns=[cols])
                model = LogisticRegression(solver='liblinear', random_state=0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)


                plt.title("Confusion Matrix (Logistic)")
                plt.figure(figsize=(10,7))
                sns.heatmap(cm, annot=True)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')

                st.pyplot(plt)
                # st.caption(f":green[Độ chính xác mô hình là: {accuracy}]")
                st.markdown(f"<h6 style='text-align: center;color: green'>Độ chính xác mô hình là:{accuracy}</h6>", unsafe_allow_html=True)

                
            elif select_model == "Hồi quy KNN":
                # st.header(":red[Biểu đồ hồi quy KNN]")
                st.markdown("<h3 style='text-align: center;color: red'>Biểu đồ hồi quy KNN</h3>", unsafe_allow_html=True)

                model = KNeighborsClassifier(n_neighbors=5)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                plt.title("Confusion Matrix (KNN)")
                plt.figure(figsize=(10,7))
                sns.heatmap(cm, annot=True)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')

                st.pyplot(plt)
                # st.caption(f":green[Độ chính xác mô hình là: {accuracy}]")
                st.markdown(f"<h6 style='text-align: center;color: green'>Độ chính xác mô hình là:{accuracy}</h6>", unsafe_allow_html=True)

            elif select_model == "DecisionTreeClassifier":
                # st.header(":red[DecisionTreeClassifier]")
                st.markdown("<h3 style='text-align: center;color: red'>DecisionTreeClassifier</h3>", unsafe_allow_html=True)


                model = DecisionTreeClassifier(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
            
                plt.figure(figsize=(20, 10))
                plot_tree(model, filled=True, feature_names=input_vars, class_names=np.unique(y).astype(str), rounded=True)
                plt.show()

                st.pyplot(plt)

            elif select_model == "DecisionTreeRegressor":
                # st.header(":red[DecisionTreeRegressor]")
                st.markdown("<h3 style='text-align: center;color: red'>DecisionTreeRegressor</h3>", unsafe_allow_html=True)

                model = DecisionTreeRegressor(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                # results.append(f"Decision Tree MSE: {mse}")
                
                plt.figure(figsize=(20, 10))
                plot_tree(model, filled=True, feature_names=input_vars, rounded=True)
                plt.show() 

                st.pyplot(plt)

            elif select_model == "Random Forest":
                # st.header(":red[Random Forest]")
                st.markdown("<h3 style='text-align: center;color: red'>Random Forest</h3>", unsafe_allow_html=True)

                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test) 
                # Tính Confusion Matrix và Classification Report
                cm = confusion_matrix(y_test, y_pred)
                cr = classification_report(y_test, y_pred)

                # Vẽ Confusion Matrix
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                fig, ax = plt.subplots(figsize=(10, 10))
                disp.plot(ax=ax)
                plt.show()
                # results.append(f"Random Forest Confusion Matrix:\n{cm}\nClassification Report:\n{cr}")

                st.pyplot(plt)

            else:
                st.write(":red[Vui lòng chọn mô hình thực hiện]") 

            # st.write(X)
            # st.write(y)
            
    else:
        st.caption("=> Tải dataset lên để xem chi tiết!!!")
if __name__ == "__main__":
    main()