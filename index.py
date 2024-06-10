import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.tree import export_graphviz
import graphviz
import pydotplus
import pydot
import seaborn as sns
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
from streamlit_option_menu import option_menu

import preprocess

# Function to convert categorical features to numerical
def convert_to_number(df, column_name):
    df[column_name] = pd.factorize(df[column_name])[0]
    return df


# plot
def display_plot(y_test, y_pred, model_choice):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{model_choice}")
    st.pyplot(fig)


# Main function to render Streamlit app
def main():
    st.set_page_config(page_title="Ứng dụng phân tích dữ liệu", layout="wide")

    st.title(":green[Ứng dụng phân tích dữ liệu 📁]")

    with st.sidebar:
        st.image("logo.png", use_column_width=True)
        st.title(":green[Ứng dụng phân tích của Nhóm 3]")
        st.header("Tải lên file CSV để bắt đầu phân tích 👇")
        uploaded_file = st.file_uploader("", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.write("Tên file:", uploaded_file.name)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Thông tin dữ liệu:", df.describe())
            st.write("Kích thước dữ liệu:", df.shape)
        with col2:
            # Show the dataframe
            st.write("Dữ liệu:")
            st.dataframe(df)

        st.markdown("""---""")

        st.title(':green[Danh sách các loại biến]📋 ')
        with st.expander("CÁC LOẠI BIẾN "):
            marks1, marks2, marks3 = st.columns(3, gap='large')
            with marks1:
                st.info('Biến số ', icon="🔢")
                numerical_vars = df.select_dtypes(include='number').columns
                st.write(numerical_vars)
            with marks2:
                st.info('Biến phân loại', icon="🔡")
                categorical_vars = df.select_dtypes(include='object').columns
                st.write(categorical_vars)
            with marks3:
                st.info('Giá trị rỗng', icon="⛔")
                st.write(df.isnull().sum())

        st.markdown("""---""")

        st.title(':red[Tiền xử lý dữ liệu]  🔧')
        cot1,cot2,cot3 = st.columns(3)
        with cot1:
            st.info('XÓA COLUMN')
            column_to_delete = st.selectbox("Chọn cột để xóa", df.columns)
            if st.button("Ấn vào đây để xóa cột"):
                preprocess.delete_column(df, column_to_delete)
        with cot2:
            st.info('KIỂM TRA VÀ XỬ LÝ DUPLICATE DỮ LIỆU')
            if st.button("Xóa hàng trùng lặp"):
                preprocess.remove_duplicates(df)

        with cot3:
            st.info('CHUYỂN ĐỔI BIẾN PHÂN LOẠI THÀNH BIẾN SỐ')
            if st.button("Chuyển đổi biến phân loại thành số"):
                preprocess.convert_categorical_to_numeric(df)


        st.markdown("""---""")

        cot4, cot5 = st.columns(2)

        with cot4:
            st.info(':green[ĐỔI TÊN COLUMN]')
            old_column_name = st.selectbox("Chọn cột cần đổi tên", df.columns)
            new_column_name = st.text_input("Nhập tên mới cho cột", value=old_column_name)
            if st.button("Ấn vào đây để  đổi tên cột"):
                preprocess.rename_column(df, old_column_name, new_column_name)

        with cot5:
            st.info(':green[XỬ LÝ GIÁ TRỊ BỊ NULL]')
            column_to_replace = st.selectbox("Chọn cột để thay thế giá trị null", df.columns)
            replace_method = st.selectbox("Phương pháp thay thế",
                                          ["Xóa", "Trung bình", "Trung vị", "Tùy chỉnh"])
            if st.button("Thay thế giá trị null"):
                if replace_method == "Xóa":
                    df.dropna(subset=[column_to_replace], inplace=True)
                    st.success(f"Đã xóa hàng chứa giá trị null trong cột '{column_to_replace}' thành công.")
                elif replace_method == "Trung bình":
                    preprocess.replace_null_with_mean(df, column_to_replace)
                elif replace_method == "Trung vị":
                    preprocess.replace_null_with_median(df, column_to_replace)
                elif replace_method == "Tùy chỉnh":
                    custom_value = st.text_input("Nhập giá trị tùy chỉnh", value="")
                    preprocess.replace_null_with_custom_value(df, column_to_replace, custom_value)

        st.subheader("Dữ liệu sau chỉnh sửa")
        st.dataframe(df)

        if st.button("🔽 Lưu tệp tại đây"):
            preprocess.save_file(df, "edited_files.csv")

        st.markdown("""---""")


        st.title(':red[Xử lý dữ liệu ở đây nhé!]  💖')
        col3, col4 = st.columns(2)
        with col3:
            features = st.multiselect("Chọn biến x:", df.columns)
        with col4:
            target = st.selectbox("Chọn biến y:", df.columns)

        if features and target:
            # Convert categorical features to numerical
            df_copy = df.copy()
            for col in features:
                if df_copy[col].dtype == 'object':
                    df_copy = convert_to_number(df_copy, col)
            if df_copy[target].dtype == 'object':
                df_copy = convert_to_number(df_copy, target)

            # Model selection
            model_choice = st.selectbox("Chọn Mô Hình:",
                                        ["Hồi Quy Logistic", "Hồi Quy Tuyến Tính", "KNN", "Decision Tree",
                                         "Random Forest"])

            if st.button("Chạy Mô Hình"):

                if model_choice == "Hồi Quy Logistic":
                    logistci_regression_model(st, df_copy, features, target, model_choice)
                elif model_choice == "Hồi Quy Tuyến Tính":
                    linear_regression_model(st, df_copy, features, target, model_choice)
                elif model_choice == "KNN":
                    knn_model(st, df_copy, df, features, target, model_choice)
                elif model_choice == "Decision Tree":
                    decision_tree_model(st, df_copy, df, features, target, model_choice)
                elif model_choice == "Random Forest":
                    random_forest_model(st, df_copy, df, features, target, model_choice)

            # Plot options
            plot_type = st.selectbox("Chọn loại biểu đồ:",
                                     ["Biểu Đồ Cột", "Biểu Đồ Đường", "Biểu Đồ Phân Phối", "Biểu Đồ Hình Tròn"])
            if st.button("Hiển Thị Biểu Đồ"):

                left, right = st.columns(2)
                plot_title = f"{plot_type} - {features[0]} vs {target}"
                title_html = f"<h3 style='text-align: center; color: green;'>{plot_title}</h3>"
                with left:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    st.markdown(title_html, unsafe_allow_html=True)

                    if plot_type == "Biểu Đồ Cột":
                        df.plot(kind='bar', x=features[0], y=target, ax=ax)
                    elif plot_type == "Biểu Đồ Đường":
                        df.plot(kind='line', x=features[0], y=target, ax=ax)
                    elif plot_type == "Biểu Đồ Phân Phối":
                        df.plot(kind='scatter', x=features[0], y=target, ax=ax)
                    elif plot_type == "Biểu Đồ Hình Tròn":
                        if df[target].dtype == "object":
                            df[target].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
                        else:
                            df[features[0]].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)

                    st.pyplot(fig)


# logistic_regression
def logistci_regression_model(st, df, features, target, model_choice):
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)
    st.write(f"Độ chính xác của mô hình {model_choice}: {accuracy}")


# linear_regrssion
def linear_regression_model(st, df, features, target, model_choice):
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    st.write(f"Độ chính xác của mô hình {model_choice}: {accuracy}")
    # Vẽ biểu đồ tương quan giữa y_pred và y_train
    display_plot(y_test, y_pred, model_choice)


# knn
def knn_model(st, df_copy, df, features, target, model_choice):
    X_train, X_test, y_train, y_test = train_test_split(df_copy[features], df_copy[target], test_size=0.2,
                                                        random_state=42)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    plt.title("Confusion Matrix (KNN)")
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    st.pyplot(plt)
    st.success(f"Độ chính xác của mô hình {model_choice}: {accuracy}")


# decision tree
def decision_tree_model(st, df_copy, df, features, target, model_choice):
    if pd.api.types.is_numeric_dtype(df[target]):  # regression
        X_train, X_test, y_train, y_test = train_test_split(df_copy[features], df_copy[target], test_size=0.2,
                                                            random_state=42)
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        display_plot(y_test, y_pred, model_choice)
    else:
        # Classification
        df_copy_1 = df.copy()
        for col in features:
            if df_copy_1[col].dtype == 'object':
                df_copy_1 = convert_to_number(df_copy_1, col)

        X_train, X_test, y_train, y_test = train_test_split(df_copy_1[features], df_copy_1[target], test_size=0.2,
                                                            random_state=42)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        st.write(cm)
        dot_data = StringIO()
        export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                        feature_names=features, class_names=model.classes_)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        st.graphviz_chart(graph.to_string())

    st.write(f"Độ chính xác của mô hình {model_choice}: {str(accuracy)}")


# random forest
def random_forest_model(st, df_copy, df, features, target, model_choice):
    if pd.api.types.is_numeric_dtype(df[target]):  # regression
        X_train, X_test, y_train, y_test = train_test_split(df_copy[features], df_copy[target], test_size=0.2,
                                                            random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        display_plot(y_test, y_pred, model_choice)
    else:
        # Classification
        df_copy_1 = df.copy()
        for col in features:
            if df_copy_1[col].dtype == 'object':
                df_copy_1 = convert_to_number(df_copy_1, col)

        X_train, X_test, y_train, y_test = train_test_split(df_copy_1[features], df_copy_1[target], test_size=0.2,
                                                            random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        st.write(cm)
    st.write(f"Độ chính xác của mô hình {model_choice}: {str(accuracy)}")


if __name__ == "__main__":
    main()