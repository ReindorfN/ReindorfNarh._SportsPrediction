
#Library imports
import joblib
import pandas as pd
import streamlit as stl


def welcome():
    return "Welcome to the FIFA Player Rating Predictor!!"


# Error handling when loading model
file_path = "best_model_predictor.pkl"
try:
    best_model = joblib.load(file_path)
    stl.success("Model loaded successfully!")
except Exception as e:
    stl.error(f"Model loading error!!: {str(e)}")
    stl.stop()

# Important features used to train the best model.
features = ['skill_long_passing',
            'attacking_crossing',
            'power_long_shots',
            'age',
            'mentality_composure',
            'power_shot_power',
            'skill_curve',
            'mentality_vision',
            'movement_reactions',
            'potential',
            'mentality_aggression',
            'skill_ball_control',
            'attacking_short_passing'
            ]

# Function for processing the input data
def preprocess(input):
    input_df = pd.DataFrame([input])
    # Ensure the input data is ordered according to the common_features list
    input_df = input_df[features]
    return input_df


# Function for predicting the player's rating
def prediction(input):
    processed_input = preprocess(input)
    prediction = best_model.predict(processed_input)
    return best_model.predict(processed_input)[0]



def main():
    stl.title("FIFA Player Rating Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit FIFA Player Rating Predictor ML App </h2>
    </div>
    """
    stl.markdown(html_temp,unsafe_allow_html=True)

    welcome()
    stl.title('FIFA Player Rating Predictor')

    # input for the features
    skill_long_passing = stl.slider('Skill Long Passing', 0, 100)
    attacking_crossing = stl.slider('Attacking Crossing', 0, 100)
    power_long_shots = stl.slider('Power Long Shots', 0, 100)
    age = stl.number_input('Age', min_value= 15, max_value= 50)
    mentality_composure = stl.slider('Mentality Composure', 0, 100)
    power_shot_power = stl.slider('Power Shot Power', 0, 100)
    skill_curve = stl.slider('Skill Curve', 0, 100)
    mentality_vision = stl.slider('Mentality Vision', 0, 100)
    movement_reactions = stl.slider('Movement Reactions', 0, 100)
    potential = stl.slider('Potential', 0, 100)
    mentality_aggression = stl.slider('Mentality Aggression', 0, 100)
    skill_ball_control = stl.slider('Skill Ball Control', 0, 100)
    attacking_short_passing = stl.slider('Attacking Short Passing', 0, 100)


    # prediction button
    if stl.button('Predict Player Rating'):
        inputs = {
            'skill_long_passing' : skill_long_passing,
            'attacking_crossing': attacking_crossing,
            'power_long_shots' : power_long_shots,
            'age' : age,
            'mentality_composure' : mentality_composure,
            'power_shot_power' : power_shot_power,
            'skill_curve' : skill_curve,
            'mentality_vision' : mentality_vision,
            'movement_reactions' : movement_reactions,
            'potential' : potential,
            'mentality_aggression' : mentality_aggression,
            'skill_ball_control' : skill_ball_control,
            'attacking_short_passing' : attacking_short_passing,  
        }

        # prediction
        try:
            result= prediction(inputs)
            stl.success(f'Player rating prediction: {result:.2f}')
        except ValueError as e:
            stl.error(f"Prediction error: {str(e)}")

    if stl.button("About"):
        stl.text("Lets LEarn")
        stl.text("Built with Streamlit")

if __name__=='__main__':
    main()
