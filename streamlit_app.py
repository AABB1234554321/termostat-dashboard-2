import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import numpy as np

# --- Streamlit App Title and Description ---
st.title("Smart Thermostat Simulation with Q-learning")
st.write("This simulation demonstrates how a thermostat learns to control room temperature using reinforcement learning (Q-learning).")

# --- Input Parameters (Sliders and Number Inputs) ---
room_temperature = st.number_input("Initial Room Temperature (°C)", min_value=10, max_value=30, value=19)
outside_temperature = st.number_input("Outside Temperature (°C)", min_value=0, max_value=40, value=10)
thermostat_setting = st.number_input("Thermostat Setting (°C)", min_value=15, max_value=25, value=20)
heater_power = st.slider("Heater Power (°C/minute)", min_value=0.1, max_value=0.5, value=0.3)
heat_loss = st.slider("Heat Loss (°C/minute)", min_value=0.05, max_value=0.2, value=0.1)

# --- Q-learning and Simulation Parameters (Can be hidden from the user) ---
num_states = 41
num_actions = 2
q_table = np.zeros((num_states, num_actions))
learning_rate = 0.1
discount_factor = 0.95
exploration_rate = 0.1
episodes = st.number_input("Training Episodes", min_value=100, max_value=5000, value=1000)
simulation_minutes = st.number_input("Simulation Minutes", min_value=10, max_value=120, value=60)

# --- Simulation ---
# ... (Your Q-learning simulation code remains mostly the same, but put it within a function)
def run_simulation():
    # ... (Your simulation code)
    # ...
    return time_all, room_temperatures_all, outside_temperatures_all, heater_output_all, df, loss_area_data

# --- Run the Simulation When a Button is Clicked ---
if st.button("Run Simulation"):
    time_all, room_temperatures_all, outside_temperatures_all, heater_output_all, df, loss_area_data = run_simulation()

    # --- Plotting with Streamlit ---
    # ... (Matplotlib plot)
    st.pyplot(plt) 

    # --- Interactive Altair Chart ---
    # ... (Your Altair Chart code)
    st.altair_Chart(Chart, use_container_width=True) 

    # --- Display Raw Data (Optional) ---
    if st.checkbox("Show Raw Data"):
        st.subheader("Simulation Data")
        st.write(df)
