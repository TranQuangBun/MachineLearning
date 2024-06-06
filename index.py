import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Function to convert categorical features to numerical
def convert_to_number(df, column_name):
    df[column_name] = pd.factorize(df[column_name])[0]
    return df

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
            with col2:
                # Show the dataframe
                st.write("Dữ liệu:")
                st.dataframe(df)

            st.title(':red[Xử lý dữ liệu ở đây nhé!]  💖')
            col3, col4 = st.columns(2)
            with col3:
                features = st.multiselect("Chọn biến x:", df.columns)
            with col4:
                target = st.selectbox("Chọn biến y:", df.columns)

            if features and target:
                # Convert categorical features to numerical
                for col in features:
                    if df[col].dtype == 'object':
                        df = convert_to_number(df, col)
                if df[target].dtype == 'object':
                    df = convert_to_number(df, target)

                X = df[features]
                y = df[target]

                # Model selection
                model_choice = st.selectbox("Chọn Mô Hình:",
                                            ["Hồi Quy Logistic", "Hồi Quy Tuyến Tính", "KNN", "Decision Tree",
                                             "Random Forest"])

                if st.button("Chạy Mô Hình"):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    if model_choice == "Hồi Quy Logistic":
                        model = LogisticRegression()
                    elif model_choice == "Hồi Quy Tuyến Tính":
                        model = LinearRegression()
                    elif model_choice == "KNN":
                        model = KNeighborsClassifier()
                    elif model_choice == "Decision Tree":
                        if pd.api.types.is_numeric_dtype(y):
                            model = DecisionTreeRegressor()
                        else:
                            model = DecisionTreeClassifier()
                    elif model_choice == "Random Forest":
                        if pd.api.types.is_numeric_dtype(y):
                            model = RandomForestClassifier(n_estimators=100)
                        else:
                            model = RandomForestClassifier(n_estimators=100)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    st.write(f"Độ chính xác của mô hình {model_choice}: {accuracy}")

                    if model_choice != "Hồi Quy Tuyến Tính":
                        cm = confusion_matrix(y_test, y_pred)
                        st.write("Confusion Matrix:")
                        st.write(cm)

                # Plot options
                plot_type = st.selectbox("Chọn loại biểu đồ:",
                                         ["Biểu Đồ Cột", "Biểu Đồ Đường", "Biểu Đồ Phân Phối", "Biểu Đồ Hình Tròn"])
                if st.button("Hiển Thị Biểu Đồ"):

                    left,right = st.columns(2)
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



if __name__ == "__main__":
    main()
