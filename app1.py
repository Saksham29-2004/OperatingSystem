import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import skfuzzy as fuzz
from flask_cors import CORS 
from skfuzzy import control as ctrl

model = load_model('CPUBurstTime.h5')

app = Flask(__name__)
CORS(app)


model = load_model('CPUBurstTime.h5')

mean_values = np.array([4500.901823, 5.815424, 11997.085578, 0.934638])

std_dev_values = np.array([727686.553286, 21.051611, 19813.482681, 1.102971])

# Define the function to calculate time quantum
def calculate_time_quantum(process_attributes):
    # Define input variables
    N = ctrl.Antecedent(np.arange(1, 1000, 1), 'N')
    ABT = ctrl.Antecedent(np.arange(0, 1000, 1), 'ABT')

    # Define output variable
    time_quantum = ctrl.Consequent(np.arange(1, 101, 0.1), 'time_quantum')

    N['low'] = fuzz.trimf(N.universe, [1, 333, 666])
    N['medium'] = fuzz.trimf(N.universe, [334, 500, 666])
    N['high'] = fuzz.trimf(N.universe, [667, 833, 999])
    ABT.automf(3)

    # Define membership functions for time quantum
    time_quantum['low'] = fuzz.trimf(time_quantum.universe, [1, 16, 33])
    time_quantum['medium'] = fuzz.trimf(time_quantum.universe, [16, 40, 66])
    time_quantum['high'] = fuzz.trimf(time_quantum.universe, [50, 82, 100])

    # Define fuzzy rules for time quantum
    rule1 = ctrl.Rule(N['high'] | ABT['poor'], time_quantum['low'])
    rule2 = ctrl.Rule(N['medium'] | ABT['average'], time_quantum['medium'])
    rule3 = ctrl.Rule(N['low'] | ABT['good'], time_quantum['high'])

    # Create fuzzy control system for time quantum
    time_quantum_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    time_quantum_sim = ctrl.ControlSystemSimulation(time_quantum_ctrl)

    # Set input values
    N_val = process_attributes['N']
    ABT_val = process_attributes['ABT']
    time_quantum_sim.input['N'] = N_val
    time_quantum_sim.input['ABT'] = ABT_val

    # Compute time quantum
    time_quantum_sim.compute()
    return time_quantum_sim.output['time_quantum']


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    inputs = np.array([data['submittime'], data['reqmemory'], data['nprocs'], data['usednprocs']], dtype=float)
    inputs_normalized = (inputs - mean_values) / std_dev_values

    predictions = model.predict(np.array([inputs_normalized]))

    prediction_result = float(predictions[0][0])  
    
    return jsonify({'prediction': prediction_result})

@app.route('/fuzzy-control', methods=['POST'])
def fuzzy_control():
    # Get data from the request
    process_attributes = request.json
    
    # Call the fuzzy logic control system function
    time_quantum = calculate_time_quantum(process_attributes)

    # Return the time quantum as the API response
    return jsonify({'time_quantum': time_quantum})

if __name__ == '__main__':
    app.run(debug=True)
