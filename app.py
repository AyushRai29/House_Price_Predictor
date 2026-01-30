import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


data = pd.read_csv("Housing.csv")

num_columns = data.select_dtypes("number").columns

data[num_columns] = data[num_columns].fillna(data[num_columns].median())

cat_columns = data.select_dtypes("object").columns

data = pd.get_dummies(data, columns=cat_columns, drop_first=True)

X = data.drop("price", axis=1)
Y = data["price"]

X_train, X_test, Y_train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

st.title("House Price Predictor", text_alignment= "center")
st.write("Enter House Details to predict the Price")

input = {}

for col in X.columns:
    if X[col].dtype == "float64" or X[col].dtype == "int64":
        input[col] = st.number_input(col, float(X[col].min()), float(X[col].max()))
    else:
        input[col] = 0

input_df = pd.DataFrame([input])

input_df = input_df.reindex(columns=X.columns, fill_value=0)

if st.button("Predict Price"):
    predict = model.predict(input_df)[0]
    st.success(f"Estimated House Price: ${predict:,.2f}")