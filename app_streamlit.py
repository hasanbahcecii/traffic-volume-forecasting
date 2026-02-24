import streamlit as st  # web interface 
import requests         # send fastapi and http requests

# header and explanation
st.title("Traffic Volume Prediction - GRU")
st.markdown("""
You can send a prediction request to the FastAPI service by entering 24-hour data through this interface.
""")

# 24 hours input data 
sequence_input = []

# start a form block
with st.form("manual_input_form"):
    st.subheader("24 Hours Input Data")

    # input for each hour
    for i in range(24):
        st.markdown(f"**🕙️ Hour {i}**")

        # 7 columns grid
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

        # input and default values
        temp = col1.number_input("Temp (K)", value=290.0, min_value=270.0, max_value=310.0, step=0.1, key=f"temp_{i}")
        rain = col2.number_input("Rain (mm)", value=0.0, min_value=0.0, max_value=100.0, step=0.1, key=f"rain_{i}")
        snow = col3.number_input("Snow (mm)", value=0.0, min_value=0.0, max_value=100.0, step=0.1, key=f"snow_{i}")
        clouds = col4.slider("Clouds (%)", value=40, min_value=0, max_value=100, step=1, key=f"clouds_{i}")
        hour = i  # 0 - 23 auto
        dayofweek = col6.selectbox("Gün", options=list(range(7)), index=1, key=f"day_{i}")
        month = col7.selectbox("Ay", options=list(range(1, 13)), index=1, key=f"month_{i}")

        # add user input to the list (must be a list, not multiple args)
        sequence_input.append([temp, rain, snow, clouds, hour, dayofweek, month])

    # submit button
    submitted = st.form_submit_button("Predict")

# send request to FastAPI if submitted
if submitted:
    # FastAPI endpoint
    url = "http://localhost:8000/predict"  

    payload = {"sequence": sequence_input}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Traffic Volume: {result['predicted_traffic_volume']:.2f}  vehicle/hour")
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
