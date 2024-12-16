from flask import Flask, render_template, request
import pandas as pd
import re
import numpy as np
import math
import statistics
import joblib
import pickle

# Load train_columns from the file
with open('models/train_columns.pkl', 'rb') as f:
    train_columns = pickle.load(f)


app = Flask(__name__)

# Read the atomic radius dataset
df_atomic_radius = pd.read_csv('data/allfeatures final.csv', index_col='symbol')

# Function to extract elements and their values from a chemical formula
def extract_elements_and_values(formula):
    matches = re.findall(r'([A-Z][a-z]*)([0-9.+-]*)', formula)
    elements_dict = {element: float(value) if value else 1.0 for element, value in matches}
    return elements_dict

# Define functions to calculate various features
def calculate_weighted_mean(element_values, variable):
    total_value = sum(element_values.values())
    x_values = {element: element_values[element] / total_value for element in element_values}
    weighted_mean = sum(x * df_atomic_radius.loc[element, variable] for element, x in x_values.items())
    return weighted_mean

def calculate_range(element_values, variable):
    values = [df_atomic_radius.loc[element, variable] for element in element_values]
    variable_range = max(values) - min(values)
    return variable_range

def calculate_weighted_std_dev(element_values, variable):
    total_value = sum(element_values.values())
    x_values = {element: element_values[element] / total_value for element in element_values}
    weighted_mean = sum(x * df_atomic_radius.loc[element, variable] for element, x in x_values.items())

    weighted_std_dev = np.sqrt(sum(x * (df_atomic_radius.loc[element, variable] - weighted_mean)**2 for element, x in x_values.items()))

    return weighted_std_dev

def calculate_max(element_values, variable):
    values = [df_atomic_radius.loc[element, variable] for element in element_values]
    variable_max = max(values)
    return variable_max

def calculate_min(element_values, variable):
    values = [df_atomic_radius.loc[element, variable] for element in element_values]
    variable_min = min(values)
    return variable_min

def calculate_mode(element_values, variable):
    values = [df_atomic_radius.loc[element, variable] for element in element_values]
    variable_mode = statistics.mode(values)
    return variable_mode

# Function to extract element values from a material string
def extract_element_values(material):
    # Initialize a dictionary to store element values
    element_values = {}

    # Use regular expression to find element symbols and values
    pattern = re.compile(r'([A-Z][a-z]*)([+-]?\d*\.?\d+)?')
    matches = pattern.findall(material)

    # Iterate through matches and update the dictionary
    for match in matches:
        element, value = match
        value = float(value) if value else 1.0
        element_values[element] = value

    return element_values

# Read the thermal conductivity values from the allfeatures.csv
thermal_conductivity_df = pd.read_csv('data/allfeatures.csv')

# Create a dictionary from the dataframe where symbol is the key and thermal conductivity is the value
thermal_conductivity_values = dict(zip(thermal_conductivity_df['symbol'],
                                       thermal_conductivity_df['thermal conductivity']))

# Read the material features CSV into DataFrame
material_df = pd.read_csv('data/anemanda - Copy.csv')

# Read the element dataset
element_df = pd.read_csv('data/anemanda2.csv')

# Create a dictionary mapping elements to their atomic weights
atomic_weights = dict(zip(element_df['element'], element_df['atomic weight']))

def sum_related_atomic_weights(material):
    # Regular expression to match elements in the material string
    element_pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'

    # Find all elements in the material string
    elements = re.findall(element_pattern, material)

    # Initialize variables to store the sum of related counts and atomic weights
    sum_related_counts = 0
    sum_related_weights = 0

    # Calculate the sum of related counts and atomic weights
    for element, count in elements:
        if element in atomic_weights:
            atomic_weight = atomic_weights[element]
            # Update the count part to handle cases where count is not provided
            if count:
                sum_related_counts += float(count)
            else:
                sum_related_counts += 1  # Set count to 1 if not provided
            sum_related_weights += atomic_weight

    return len(elements), sum_related_counts, sum_related_weights


# Load the trained models
loaded_models = joblib.load('models/final_models.pkl')
rf_classifier = loaded_models['rf_classifier']
lgbm_regressor = loaded_models['lgbm_regressor']

# Function to predict critical temperature
def predict_critical_temp(features):
    # Assuming 'features' is a DataFrame containing the derived features
    # Prepare features for prediction
    X_pred = features  # Drop 'material' column if present
    
    # Predict temperature class using the trained Random Forest Classifier
    y_pred_class = rf_classifier.predict(X_pred)
    
    # Predict critical temperature using the trained LightGBM Regressor
    y_pred_critical_temp = lgbm_regressor.predict(X_pred)
    
    # Set predicted critical temperature as 0 where the predicted temperature class is 0
    y_pred_critical_temp[y_pred_class == 0] = 0
    
    return y_pred_critical_temp[0]  # Assuming only one prediction is made

@app.route('/', methods=['GET', 'POST'])
def index():
    material = ''
    pressure = ''
    predicted_critical_temp = 'predicted value(K)'  # Initialize predicted_critical_temp
    if request.method == 'POST':
        material = request.form['material']
        pressure = request.form.get('pressure', '')  # Handle the case where pressure might be empty

        
        # Extract element values from the material string
        element_values = extract_element_values(material)

        # Create a dictionary to store calculated properties
        properties = {'material': material}

        # Calculate properties for the given material
        for i in range(9):
            element = f'C{i+1}'
            if i < len(element_values):
                element_symbol = list(element_values.keys())[i]
                thermal_conductivity_value = thermal_conductivity_values.get(element_symbol, 0)
                properties[element] = element_values.get(element_symbol, 0) * thermal_conductivity_value
            else:
                properties[element] = 0

        # Extract elements from the material composition
        composition = extract_element_values(material)

        # Define specific elements of interest
        specific_elements = ['Cu', 'O']

        # Initialize values to zero
        element_weights = {element: 0 for element in specific_elements}
        element_specific_heats = {element: 0 for element in specific_elements}

        # Calculate the element weights and specific heats
        for element in specific_elements:
            if element in composition:
                total_formula_weight = sum(material_df.loc[material_df['symbol'] == el, 'atomic weight'].values[0] * count for el, count in composition.items())
                total_formula_specific_heat = sum(material_df.loc[material_df['symbol'] == el, 'specific heat'].values[0] * count for el, count in composition.items())

                # Calculate the element weight
                element_weights[element] = 100 * (composition[element] * material_df.loc[material_df['symbol'] == element, 'atomic weight'].values[0]) / total_formula_weight

                # Calculate the element specific heat
                element_specific_heats[element] = 100 * (composition[element] * material_df.loc[material_df['symbol'] == element, 'specific heat'].values[0]) / total_formula_specific_heat

        # Calculate the sum of related counts, atomic weights, and number of elements for the material
        num_elements, sum_counts, sum_weights = sum_related_atomic_weights(material)

        # Add the additional features to the properties dictionary
        properties['Number of Elements'] = num_elements
        properties['Sum of Related Counts'] = sum_counts
        properties['Sum of Related Weights'] = sum_weights

        # Add the element weights and specific heats to the properties dictionary
        properties.update({f'{element} Weight': weight for element, weight in element_weights.items()})
        properties.update({f'{element} Specific Heat': specific_heat for element, specific_heat in element_specific_heats.items()})

        # Ensure 'material' column is of string type
        material = str(material)

        # Extract elements and their values from the given material
        element_values = extract_elements_and_values(material)

        # Calculate features for the given material
        calculated_values = {'material': material}
        for variable in df_atomic_radius.columns:
            calculated_values[f'weighted_mean_{variable}'] = calculate_weighted_mean(element_values, variable)
            calculated_values[f'{variable}_range'] = calculate_range(element_values, variable)
            calculated_values[f'weighted_std_dev_{variable}'] = calculate_weighted_std_dev(element_values, variable)
            calculated_values[f'{variable}_max'] = calculate_max(element_values, variable)
            calculated_values[f'{variable}_min'] = calculate_min(element_values, variable)
            calculated_values[f'{variable}_mode'] = calculate_mode(element_values, variable)

        # Update properties dictionary with calculated values
        properties.update(calculated_values)

        # Load selected features from the CSV file
        importance_df1 = pd.read_csv('data/aaa.csv')
        selected_features = importance_df1.head(79)['Feature'].tolist()

        # Remove 'pressure' if present in the selected features list
        if 'pressure' in selected_features:
            selected_features.remove('pressure')
       
        # Filter properties dictionary to keep only selected features
        properties_selected = {key: value for key, value in properties.items() if key in selected_features}

        # Create a DataFrame from the selected properties
        df_properties = pd.DataFrame([properties_selected])

        # Include the pressure variable in df_properties as another column
        df_properties['pressure'] = int(request.form['pressure']) if 'pressure' in request.form else 0

        # Reorder columns in df_properties according to train_columns
        df_properties = df_properties[train_columns]

        # Predict the critical temperature using the trained model
        predicted_critical_temp = predict_critical_temp(df_properties)

        # Pass the number of columns along with properties DataFrame
        num_columns = len(df_properties.columns)

        # Render template with results including predicted critical temperature
        #return render_template('result.html', num_columns=num_columns, properties=df_properties.to_html(index=False), predicted_critical_temp=predicted_critical_temp)

    # Render the form template with the predicted critical temperature
    return render_template('index.html', material=material, pressure=pressure, predicted_critical_temp=predicted_critical_temp)

if __name__ == '__main__':
    app.run(debug=True)
