import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data=pd.read_csv("data.csv")
x=data[["load"]]
y=data[["extension"]]

plt.scatter(x,y)
plt.xlabel("Load")
plt.ylabel("Extension")
plt.title("Load vs Extension")
plt.show()

model=LinearRegression()
model.fit(x,y)
print("coefficient:",model.coef_)
print("intercept:",model.intercept_)
input_load=float(input("Enter the load to predict the extension:"))
pred_ext=model.predict([[input_load]])
print("predicted extension for load :",input_load,"is :",pred_ext[0][0])