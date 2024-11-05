import streamlit as st
import pickle
import numpy as np
import time


from sklearn.preprocessing import StandardScaler

# Load the pickle files
with open('xgb1_machine_failure.pkl', 'rb') as f:
    machine_failure_model = pickle.load(f)

with open('xgb2_machine_failure_type.pkl', 'rb') as f:
    failure_type_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the Streamlit app
def main():
    st.title('Predective Maintenance using Machine Learning')

    st.header('Input Features')
    
    # Use columns to create a two-card layout
    col1, col2 = st.columns(2)
    
    with col1:
       
        machine_type = st.selectbox('Type', ['L', 'M', 'H'])  # Example types
        air_temp = st.number_input('Air Temperature', step=0.1, format="%.1f")
        process_temp = st.number_input('Process Temperature', step=0.1, format="%.1f")
        
    with col2:
        
        rotational_speed = st.number_input('Rotational Speed', step=1)
        torque = st.number_input('Torque', step=0.1, format="%.1f")
        tool_wear = st.number_input('Tool Wear', step=0.1, format="%.1f")
    
    machine_name = st.text_input('Machine Name',placeholder="Enter the machine name")

    # Convert machine_type to numeric if necessary
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    machine_type_numeric = type_mapping[machine_type]

    # Combine the input features
    numeric_features = np.array([[air_temp, process_temp, rotational_speed, torque, tool_wear]])
    numeric_features_scaled = scaler.transform(numeric_features)
    input_features = np.concatenate([[machine_type_numeric], numeric_features_scaled[0]])

    if st.button('Predict'):
        # Predict machine failure
        machine_failure_prediction = machine_failure_model.predict([input_features])

        # Predict type of failure
        failure_type_prediction = failure_type_model.predict([input_features])

        # Display the results
        st.subheader('Prediction Results')
        if machine_failure_prediction[0] == 0:
            st.success(f'The machine type is: {machine_type}')
            st.success(f'The machine name is: {machine_name}')
            st.success('The machine is not expected to fail.')
        else:
            failure_types = {0: 'No Failure', 1: 'Power Failure', 2: 'Overstrain Failure', 3: 'Heat Dissipation Failure', 4: 'Tool Wear Failure'}
            failure_type = failure_types[failure_type_prediction[0]]
            
            st.error(f'The machine type is: {machine_type}')
            st.error(f'The machine name is: {machine_name}')
            st.error('The machine is likely to fail.')
            st.error(f'Type of failure predicted: {failure_type}')
            st.markdown(
                            """
                            <style>
                                div[data-testid=toastContainer] {
                                        padding: 50px 10px 10px 10px;
                                        align-items: end;
                                        position: sticky;
                                      
                                    }
                                
                                    div[data-testid=stToast] {
                                        padding: 10px 20px 10px 10px;
                                   
                                        width: 50%;
                                        foreground-colour: white;
                                    }
                                    
                                    [data-testid=toastContainer] [data-testid=stMarkdownContainer] > p {
                                        font-size: 1.1rem;
                                        font-weight: 400;
                                        padding: 10px 10px 10px 40px;
                                        text-indent: -1.7em;
                                       
                                    }
                            </style>
                            """, unsafe_allow_html=True)

            
            time.sleep(2)
            # Show alert message if failure is predicted
            if failure_type != 'No Failure':
                st.toast(f'⚠ Alert: The machine is likely to fail due to {failure_type}.')
                time.sleep(.5)
                st.toast('A notification 🔔 has been sent to the maintenance team regarding the failure')
           
            # if failure_type != 'No Failure':
            #     st.markdown(f'<div style="position: fixed; top: 100px; right: 200px; left: 200px; bottom:100px;text-align: center; background-color: #f8d7da; padding: 100px; border-radius: 5px; font-size: 15px;">⚠: The machine is likely to fail due to {failure_type}.<br> A notification 🔔 has been sent to the maintenance team.</div>', unsafe_allow_html=True)
            #     time.sleep(5)  # Sleep for 2 seconds to display the message
            #     st.experimental_rerun()

            

           
            
    
if __name__ == '__main__':
    main()
