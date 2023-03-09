import pickle
import streamlit as st
import sklearn



def predictShoeSize(height):
    predicted_shoe_size = [[999]]
    # Load the ML model
    # filename = 'sys-304/shoe.size.prediction.model.pkl'
    # with open(filename, 'rb') as file:
    #     loaded.model = pickle.load(file)

    loaded_model = pickle.load(open('shoe_size_prediction_model.pkl','rb'))


    # Use the loaded model to make predictions
    new_X = [[height]]
    predicted_shoe_size = loaded_model.predict(new_X)
    print(predicted_shoe_size)

    return predicted_shoe_size

st.markdown("***")
st.markdown("# Let's Predict your Shoe Size from your Height")
st.markdown("***")

height_row, space, shoe_size_row = st.columns([2,1,2])

with height_row:
    st.image("https://drive.google.com/uc?id=17A15LbY0ldA3uP3tDG-mAOBtZAeW6OJ7", use_column_width=True) 

    

with shoe_size_row:
    height = st.number_input('Insert your height (cm):')
    predicted_shoe_size = predictShoeSize(height)
    st.markdown("  ")
    st.markdown("  ")
    st.markdown("Your shoe size is:")
    st.markdown("# " + str(round(predicted_shoe_size[0][0],2)))
    st.markdown("  ")
    st.markdown("  ")




st.markdown("***")