import numpy as np
import pickle

import streamlit as st
st.header("Taste My Wine")

#for normalization from dataset
max_vd = pickle.load(open("./models/max_val_dict.pkl", "rb"))
max_vd.pop('quality')
maxx = np.array(list(max_vd.values()))

min_vd = pickle.load(open("./models/min_val_dict.pkl", "rb"))
min_vd.pop('quality')
minn = np.array(list(min_vd.values()))


fixed_acidity = st.slider('Fixed Acidity', min_vd['fixed acidity'], max_vd['fixed acidity'], value=11.2)
volatile_acidity = st.slider('Volatile Acidity', min_vd['volatile acidity'], max_vd['volatile acidity'], value=0.28)
citric_acid = st.slider('Citric Acid', min_vd['citric acid'], max_vd['citric acid'], value=0.56)
residual_sugar = st.slider('Residual Sugar', min_vd['residual sugar'], max_vd['residual sugar'], value=1.9)
chlorides = st.slider('Chlorides', min_vd['chlorides'], max_vd['chlorides'], value=0.075)
free_so2 = st.slider('Free Sulphur Dioxide', min_vd['free sulfur dioxide'], max_vd['free sulfur dioxide'], value=17.0)
total_so2 = st.slider('Total Sulfur Dioxide', min_vd['total sulfur dioxide'], max_vd['total sulfur dioxide'], value=60.0)
density = st.slider('Density', min_vd['density'], max_vd['density'], value=0.998)
pH = st.slider('pH', min_vd['pH'], max_vd['pH'], value=3.16)
sulphates = st.slider('Sulphates', min_vd['sulphates'], max_vd['sulphates'], value=0.58)
alcohol = st.slider('Alcohol', min_vd['alcohol'], max_vd['alcohol'], value=9.8)


# all_dic = {'fixed acidity': 11.2, 'volatile acidity': 0.28, 'citric acid': 0.56,
#            'residual sugar': 1.9, 'chlorides': 0.075, 'free sulfur dioxide': 17, 
#            'total sulfur dioxide': 60, 'density': 0.998, 'pH': 3.16, 'sulphates': 0.58,
#            'alcohol': 9.8}

all_dic = {'fixed acidity': fixed_acidity, 'volatile acidity': volatile_acidity, 'citric acid': citric_acid,
           'residual sugar': residual_sugar, 'chlorides': chlorides, 'free sulfur dioxide': free_so2, 
           'total sulfur dioxide': total_so2, 'density': density, 'pH': pH, 'sulphates': sulphates,
           'alcohol': alcohol}

norm_list = np.array(list(all_dic.values()))
#-------------features-------------

#--------normalize-------
# print("value from slider",norm_list)
infer_me_this_batman = (norm_list-minn)/(maxx-minn)
# print("supposed to be normalized value", infer_me_this_batman)
# infer_me_this_batman = norm_list
# print(infer_me_this_batman)
#--------normalize-------


if st.button('Make Prediction'):
    infer_on = np.expand_dims(infer_me_this_batman, 0)
    model = pickle.load(open("./models/regressionAllFeat.pkl","rb"))
    inferred = model.predict(infer_on)
    print("final pred", np.round(np.squeeze(inferred, -1), decimals=3))
    # print("final pred", inferred)
    st.write(f"Predicted wine quality with given features is: {np.round(np.squeeze(inferred, -1), decimals=3)*100} percent.")
    # st.write(f"The inferred wine quality is: {inferred}")


