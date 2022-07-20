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


fixed_acidity = st.slider('Fixed Acidity', 0.0, max_vd['fixed acidity'], 1.0)
volatile_acidity = st.slider('Volatile Acidity', 0.0, max_vd['volatile acidity'], 1.0)
citric_acid = st.slider('Citric Acid', 0.0, max_vd['citric acid'], 1.0)
residual_sugar = st.slider('Residual Sugar', 0.0, max_vd['residual sugar'], 1.0)
chlorides = st.slider('Chlorides', 0.0, max_vd['chlorides'], 1.0)
free_so2 = st.slider('Free Sulphur Dioxide', 0.0, max_vd['free sulfur dioxide'], 1.0)
total_so2 = st.slider('Total Sulfur Dioxide', 0.0, max_vd['total sulfur dioxide'], 1.0)
density = st.slider('Density', 0.0, max_vd['density'], 1.0)
pH = st.slider('pH', 0.0, max_vd['pH'], 1.0)
sulphates = st.slider('Sulphates', 0.0, max_vd['sulphates'], 1.0)
alcohol = st.slider('Alcohol', 0.0, max_vd['alcohol'], 1.0)


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
infer_me_this_batman = (norm_list-minn)/(maxx-minn)
# print(infer_me_this_batman)
#--------normalize-------


if st.button('Make Prediction'):
    infer_on = np.expand_dims(infer_me_this_batman, 0)
    model = pickle.load(open("./models/regressionAllFeat.pkl","rb"))
    inferred = model.predict(infer_on)
    print("final pred", np.squeeze(inferred, -1))
    # print("final pred", inferred)
    st.write(f"The wine quality is: {np.squeeze(inferred, -1)}  in perceived quality")
    # st.write(f"The inferred wine quality is: {inferred}")


