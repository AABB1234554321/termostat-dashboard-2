import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from geneticalgorithm import geneticalgorithm as ga

# --- App Title and Description ---
st.title("Thermostat Simulation with Q-Learning, PID, and ON-OFF Control")
st.write("This simulation compares Q-Learning, PID control, and ON-OFF control for maintaining room temperature.")

# --- Input Parameters ---
initial_room_temperature = st.number_input("Initial Room Temperature (°C)", min_value=10.0, max_value=30.0, value=19.0)
outside_temperature = st.number_input("Outside Temperature (°C)", min_value=0.0, max_value=40.0, value=10.0)
thermostat_setting = st.number_input("Thermostat Setting (°C)", min_value=15.0, max_value=25.0, value=20.0)
heater_power = st.slider("Heater Power (°C/minute)", min_value=0.1, max_value=0.5, value=0.3)
base_heat_loss = st.slider("Base Heat Loss (°C/minute)", min_value=0.05, max_value=0.2, value=0.1)

# Q-learning Parameters
num_states = 41
num_actions = 2
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.1
episodes = st.number_input("Training Episodes (Q-Learning)", min_value=100, max_value=5000, value=1000)
q_table = np.zeros((num_states, num_actions))  # Initialize q_table

# Simulation Parameters
simulation_minutes = st.number_input("Simulation Minutes", min_value=10, max_value=1440, value=60)

# --- Helper Functions (Q-Learning) ---
def get_state(temperature):
    return int((temperature - 10) // 0.5)

def get_action(state):
    if np.random.uniform(0, 1) < exploration_rate:
        return np.random.choice(num_actions)  # Exploration
    else:
        return np.argmax(q_table[state, :])  # Exploitation

def get_reward(state, action, thermostat_setting):
    state_temp = 10 + state * 0.5
    if abs(state_temp - thermostat_setting) <= 0.5:
        return 10  # Within acceptable range
    elif action == 1 and state_temp > thermostat_setting + 0.5:  # Too hot
        return -10
    elif action == 0 and state_temp < thermostat_setting - 0.5:  # Too cold
        return -5
    else:
        return -1  # Slight penalty for not being in range

def run_q_learning_simulation(initial_room_temperature):
    global q_table 
    for episode in range(episodes):
        room_temperature = initial_room_temperature
        state = get_state(room_temperature)
        for _ in np.arange(0, simulation_minutes, 0.1):
            action = get_action(state)
            if action == 1:
                room_temperature += heater_power * 0.1
            else:
                heat_loss = base_heat_loss * (room_temperature - outside_temperature)
                room_temperature -= heat_loss * 0.1

            next_state = get_state(room_temperature)
            reward = get_reward(next_state, action, thermostat_setting)

            # Update Q-table
            q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state

    # Run one final simulation using the learned Q-table
    time = []
    room_temperatures = []
    heater_output = []

    room_temperature = initial_room_temperature
    state = get_state(room_temperature)
    for minute in np.arange(0, simulation_minutes, 0.1):
        action = np.argmax(q_table[state, :])  # Always choose the best action
        heater_output.append(action)

        if action == 1:
            room_temperature += heater_power * 0.1
        else:
            heat_loss = base_heat_loss * (room_temperature - outside_temperature)
            room_temperature -= heat_loss * 0.1

        state = get_state(room_temperature)
        time.append(minute)
        room_temperatures.append(room_temperature)

    df = pd.DataFrame({
        'Time (Minutes)': time,
        'Room Temperature (°C)': room_temperatures,
        'Heater Output': heater_output
    })

    return time, room_temperatures, heater_output, df



# --- Simulation Logic (PID) ---
def run_pid_simulation(initial_room_temperature, Kp, Ki, Kd):
    # ... (PID simulation logic remains the same)

# --- Simulation Logic (ON-OFF) ---
def run_on_off_simulation(initial_room_temperature):
    # ... (ON-OFF simulation logic remains the same)

# --- Calculate Area Between Current Temperature and Set Temperature ---
def calculate_area_between_temp(time, room_temperatures, set_temp):
    # ... (Area calculation function remains the same)

# --- Optimization Function for PID Parameters ---
def optimize_pid(params):
    # ... (PID optimization function remains the same)


# --- Main App ---
simulation_type = st.selectbox("Choose Simulation Type:", ("Q-Learning", "PID", "ON-OFF", "All"))

if st.button("Run Simulation"):
    time_q = room_temperatures_q = heater_output_q = df_q = None
    time_pid = room_temperatures_pid = heater_output_pid = df_pid = None
    time_onoff = room_temperatures_onoff = heater_output_onoff = df_onoff = None

    if simulation_type == "Q-Learning" or simulation_type == "All":
        time_q, room_temperatures_q, heater_output_q, df_q = run_q_learning_simulation(initial_room_temperature)
        area_q = calculate_area_between_temp(time_q, room_temperatures_q, thermostat_setting)
        st.write(f"Heat loss with Q-learning: {area_q:.2f} °C*minutes")

    if simulation_type == "PID" or simulation_type == "All":
        # Optimize PID parameters using Genetic Algorithm
        varbound = np.array([[0.1, 1000.0], [0.00001, 0.5], [0.001, 0.9]])
        algorithm_param = {'max_num_iteration': 100, 'population_size': 50, 'mutation_probability': 0.1, 'elit_ratio': 0.01, 'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type': 'uniform', 'max_iteration_without_improv': None}
        model = ga(function=optimize_pid, dimension=3, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
        model.run()
        best_params = model.output_dict['variable']
        Kp, Ki, Kd = best_params

        st.write(f"Optimized PID Parameters: Kp={Kp:.2f}, Ki={Ki:.5f}, Kd={Kd:.3f}")

        time_pid, room_temperatures_pid, heater_output_pid, df_pid = run_pid_simulation(initial_room_temperature, Kp, Ki, Kd)
        area_pid = calculate_area_between_temp(time_pid, room_temperatures_pid, thermostat_setting)
        st.write(f"Heat loss with PID: {area_pid:.2f} °C*minutes")

    if simulation_type == "ON-OFF" or simulation_type == "All":
        time_onoff, room_temperatures_onoff, heater_output_onoff, df_onoff = run_on_off_simulation(initial_room_temperature)
