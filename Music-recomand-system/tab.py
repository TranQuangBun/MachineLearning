import streamlit as st

# Tạo các lựa chọn trong sidebar
option = st.sidebar.radio("Chọn tab:", ["Tab 1", "Tab 2", "Tab 3"])

# Hiển thị nội dung dựa trên lựa chọn
if option == "Tab 1":
    st.header("This is tab 1")
    st.write("Content for tab 1")
elif option == "Tab 2":
    st.header("This is tab 2")
    st.write("Content for tab 2")
elif option == "Tab 3":
    st.header("This is tab 3")
    st.write("Content for tab 3")
