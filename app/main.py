import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def add_predictions(input_data):
    model = pickle.load(open(r"cancer_detection/model.pkl","rb"))
    scaler = pickle.load(open(r"cancer_detection/scaler.pkl","rb"))
    #input_data is in the form of dictionary with key value pairs, we have to convert it in a single array, for this we use numpy
    input_array = np.array(list(input_data.values())).reshape(1,-1)  # our model is supposed to take 2d arrays instead of 1d
    

    #scale them with the scaler model that we have imported, as we need to pass these values into our model and our model takes scaled values
    input_array_scaled = scaler.transform(input_array) # all comes down to zeros, this is because we have set the default values of sliders as the mean of that column
    

    #fit the scaled values in our model
    prediction = model.predict(input_array_scaled)

    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is : ")

    if prediction[0] ==0:
        st.write("<span class = 'diagnosis benign'>Benign</span>",unsafe_allow_html= True)
    else:
        st.write("<span class = 'diagnosis malicious'>Malicious</span>",unsafe_allow_html= True)  
    st.write("Probability of being benign :", model.predict_proba(input_array_scaled)[0][0])  # this returns an array with 2 elements, probab of being 0 an dprobab of being 1
    st.write("Probability of being malicious :", model.predict_proba(input_array_scaled)[0][1])    
    
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


def get_scaled_value(input_data):
    data = getCleanData()
    X = data.drop(["diagnosis"],axis=1)
    scaled_dict ={}
    
    for key,value in input_data.items():
        max_val = X[key].max()               # we can do scaling using sklearn as well
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val-min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict

def getCleanData():
    df = pd.read_csv(r"cancer_detection/cancer.csv") # should be in the same directory as the file
    df = df.drop(["Unnamed: 32","id"],axis =1) # axis = 1 means column, axis = 0 means row
    df["diagnosis"] = df["diagnosis"].map({"M":1,"B":0}) #encoding M and B to 1's and 0's to convert it into numbers

    return df

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    #create slider, each slider has a name and maximum value that it can take, export an object with col names, max values etc.,we export this using the pickle function
    # because the dataset in this case is small, we can just put the dataset into our app folder
    data =getCleanData()
    slider_labels =[
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),    # 1st one is for the label name of the slider, and the 2nd one is to get the max value of this from the data
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst")
    ]

    input_dict ={} #to store value of slider
    # here we are using slider, bu we can pretty much use any html input element, go to streamlit inputs on the web to check for more stuff
    for label,key in slider_labels:
        input_dict[key] = st.sidebar.slider(  #this slider returns the value input in the slider so we need to store it somewhere to make predictions and plot the chart
            label,
            min_value=float(0),  # we have to mak eboth min and max as same data type else it will give error
            max_value=float(data[key].max()),
            value = float(data[key].mean())  #default value

        )  #sidebar.slider done to pu slider inside the sidebar

    return input_dict

def get_radar_chart(input_data):
    # we draw a radar chart of our data using a python library called plotly, which helps us draw interactive charts rather than seaborn or matplotlib, has different value at radii, we use a javascript library having a python module
    #in the data for every parameter we have 3 columns eg for radii, we have mean, standard error and worse
    input_data = get_scaled_value(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
           input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
           input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
           input_data['fractal_dimension_mean']],
        theta=categories,
        fill='toself',
        name='mean'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
           input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
           input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
        theta=categories,
        fill='toself',
        name='standard error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']],
        theta=categories,
        fill='toself',
        name='worst'
    ))



    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )
    # as different range of values of all features so we need to scale them between 0 and 1
    #fig.show(), dont use this, streamlit uses its own functions incorporate plotly elements into application
    return fig
def main():
    # set the page config for our app
    st.set_page_config(
        page_title= "Breast Cancer Predictor",
        page_icon= "👩‍⚕️",
        layout= "wide",  # there is a default conatiner inside which all the dat is written, wide is done to make it bigger
        initial_sidebar_state= "expanded"  # we have to make this sidebar

    )
    
    with open(r"assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()),unsafe_allow_html= True)

    #adding sidebar using function
    input_data = add_sidebar() #updation  of the parameters happen in real time as if we have got the prediction, then we update the parameters, it is passed into the model real time to make new predictions i.e automatically



    with st.container(): # whatever inside this is inside this container, containers are a very structured to make your application
        st.title("Breast Cancer Predictor",)
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ") # it creates a p element in html i.e paragraph
        # ideally, it should be plugged in with the machine that is doing measurements, so the values are fed in automatically

    # no create 2 columns, one containing th chart and other containing the prediction box
    col1,col2 = st.columns([4,1]) # ratio of 2 columns as paramter

    with col1:
        radar_chart=get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)  

if __name__ == "__main__":
    main()

# run using streamlit run app/main.py    