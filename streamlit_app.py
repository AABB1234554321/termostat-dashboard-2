import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import numpy as np

# --- App Title and Description ---
st.title("Smart Thermostat Simulation with Q-learning")
st.write("This simulation demonstrates how a thermostat learns to control room temperature using reinforcement learning (Q-learning).")

# --- Input Parameters ---
room_temperature = st.number_input("Initial Room Temperature (°C)", min_value=10, max_value=30, value=19)
outside_temperature = st.number_input("Outside Temperature (°C)", min_value=0, max_value=40, value=10)
thermostat_setting = st.number_input("Thermostat Setting (°C)", min_value=15, max_value=25, value=20)
heater_power = st.slider("Heater Power (°C/minute)", min_value=0.1, max_value=0.5, value=0.3)
heat_loss = st.slider("Heat Loss (°C/minute)", min_value=0.05, max_value=0.2, value=0.1)

# --- Q-learning and Simulation Parameters ---
num_states = 41  # Temperature states (10°C to 30°C in 0.5°C increments)
num_actions = 2  # Heater off (0) or on (1)
q_table = np.zeros((num_states, num_actions))  # Initialize Q-table
learning_rate = 0.1  # Alpha
discount_factor = 0.95  # Gamma
exploration_rate = 0.1  # Epsilon
episodes = st.number_input("Training Episodes", min_value=100, max_value=5000, value=1000)
simulation_minutes = st.number_input("Simulation Minutes", min_value=10, max_value=120, value=60)


# --- Helper Functions ---
def get_state(temperature):
    """Discretize temperature into states."""
    return int((temperature - 10) // 0.5)


def get_action(state):
    """Choose an action based on the Q-table and exploration rate."""
    if np.random.uniform(0, 1) < exploration_rate:
        return np.random.choice(num_actions)  # Exploration
    else:
        return np.argmax(q_table[state, :])   # Exploitation


def get_reward(state, action):
    """Calculate reward based on state and action."""
    if abs(state - get_state(thermostat_setting)) <= 1:  # Within acceptable range
        return 10
    elif action == 1 and state > get_state(thermostat_setting + 0.5):  # Too hot
        return -10
    elif action == 0 and state < get_state(thermostat_setting - 0.5):  # Too cold
        return -5
    else:
        return 0

# --- Simulation Logic ---
def run_simulation():
    # Lists to store episode data
    time = []
    room_temperatures = []
    outside_temperatures = []
    heater_output = []

    # Q-Learning training
    for episode in range(episodes):
        room_temperature = 19  # Reset for each episode
        state = get_state(room_temperature)
        for minute in np.arange(0, simulation_minutes, 0.1):
            action = get_action(state)
            heater_output.append(action)

            # Update temperature
            if action == 1:
                room_temperature += heater_power * 0.1
            else:
                room_temperature -= heat_loss * 0.1

            next_state = get_state(room_temperature)
            reward = get_reward(next_state, action)

            # Q-Table update
            q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state

            time.append(minute)
            room_temperatures.append(room_temperature)
            outside_temperatures.append(outside_temperature)

    # Dataframes for plotting
    df = pd.DataFrame({
        'Time (Minutes)': time,
        'Room Temperature (°C)': room_temperatures,
        'Outside Temperature (°C)': outside_temperatures,
        'Heater Output': heater_output
    })

    loss_area_data = pd.DataFrame({
        'Time (Minutes)': time,
        'Lower Limit': [thermostat_setting] * len(time),
        'Upper Limit': room_temperatures
    })
    loss_area_data = loss_area_data[loss_area_data['Upper Limit'] > loss_area_data['Lower Limit']]

    return time, room_temperatures, outside_temperatures, heater_output, df, loss_area_data



# --- Run Simulation & Plot Results ---
if st.button("Run Simulation"):
    time, room_temperatures, outside_temperatures, heater_output, df, loss_area_data = run_simulation()

    # Matplotlib Plot
    plt.figure(figsize=(10, 6))
    # ... your plotting code ...
    st.pyplot(plt)

    # Altair Chart
    chart = alt.Chart(df).mark_line().encode(
        x='Time (Minutes)',
        y='Room Temperature (°C)',
        color=alt.value('blue'),
        tooltip=['Time (Minutes)', 'Room Temperature (°C)']
    )
    # ... your Altair chart code ...
    st.altair_chart(chart, use_container_width=True)

    # Option to show raw data
    if st.checkbox("Show Raw Data"):
        st.subheader("Simulation Data")
        st.write(df)
