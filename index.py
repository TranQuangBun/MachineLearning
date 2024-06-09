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
                
            row1, row2, row3 = st.columns(3)
            with row1:
                st.title('Biến số ')
                numerical_vars = df.select_dtypes(include='number').columns
                st.write(numerical_vars)
            with row2:
                st.title('Biến phân loại')
                categorical_vars = df.select_dtypes(include='object').columns
                st.write(categorical_vars)
                
            with row3:
                st.title("Giá trị rỗng")
                st.write(df.isnull().sum())
                    
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
                        
#logistic_regression
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


#linear_regrssion
def linear_regression_model(st, df, features, target, model_choice):
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)     
    y_pred = model.predict(X_test)  
    accuracy = model.score(X_test, y_test)
    
    st.write(f"Độ chính xác của mô hình {model_choice}: {accuracy}")
     # Vẽ biểu đồ tương quan giữa y_pred và y_train
    display_plot(y_test, y_pred, model_choice)
    

#knn
def knn_model(st, df_copy, df, features, target, model_choice):
    X_train, X_test, y_train, y_test = train_test_split(df_copy[features], df_copy[target], test_size=0.2, random_state=42)
    
    if pd.api.types.is_numeric_dtype(df[target]): # regression
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)         
        accuracy = model.score(X_test, y_test)
        display_plot(y_test, y_pred, model_choice)
    else: #Classification
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        st.write(cm)
        
    st.write(f"Độ chính xác của mô hình {model_choice}: {accuracy}") 
    
#decision tree
def decision_tree_model(st, df_copy, df, features, target, model_choice): 
    if pd.api.types.is_numeric_dtype(df[target]): # regression
        X_train, X_test, y_train, y_test = train_test_split(df_copy[features], df_copy[target], test_size=0.2, random_state=42)
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)    
        y_pred = model.predict(X_test)     
        accuracy = model.score(X_test, y_test)
        display_plot(y_test, y_pred, model_choice)
    else: 
        #Classification
        df_copy_1 = df.copy()
        for col in features:
            if df_copy_1[col].dtype == 'object':
                df_copy_1 = convert_to_number(df_copy_1, col)
        
        X_train, X_test, y_train, y_test = train_test_split(df_copy_1[features], df_copy_1[target], test_size=0.2, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        st.write(cm)
        dot_data = StringIO()
        export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=features, class_names=model.classes_) 
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        st.graphviz_chart(graph.to_string())
        
    st.write(f"Độ chính xác của mô hình {model_choice}: {str(accuracy)}") 
    
#random forest
def random_forest_model(st, df_copy, df, features, target, model_choice):
    if pd.api.types.is_numeric_dtype(df[target]): # regression
        X_train, X_test, y_train, y_test = train_test_split(df_copy[features], df_copy[target], test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)    
        y_pred = model.predict(X_test)     
        accuracy = model.score(X_test, y_test)
        display_plot(y_test, y_pred, model_choice)
    else: 
        #Classification
        df_copy_1 = df.copy()
        for col in features:
            if df_copy_1[col].dtype == 'object':
                df_copy_1 = convert_to_number(df_copy_1, col)
        
        X_train, X_test, y_train, y_test = train_test_split(df_copy_1[features], df_copy_1[target], test_size=0.2, random_state=42)
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
