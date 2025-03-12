import streamlit as st

st.set_page_config(layout="wide")

# if not st.experimental_user.is_logged_in:
#     if st.button("Log in with Google"):
#         st.login()
#     st.stop()

# if st.button("Log out"):
#     st.logout()

# st.markdown(f"Welcome! {st.experimental_user.name}")

pg = st.navigation(
    [
        st.Page("file_analysis.py", title="Анализ Расходов"),
    ]
)
pg.run()
