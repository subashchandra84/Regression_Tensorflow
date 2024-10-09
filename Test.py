from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset
X = [[1, 200], [2, 300], [3, 400], [4, 500], [5, 600], [6, 700]]
y = [10, 20, 30, 40, 50, 60]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

X_train_scaled1 = scaler.fit(X_train)


# Fit and transform the training data
X_train_scaled = scaler.transform(X_train)

# Transform the test data (do NOT fit again)
X_test_scaled = scaler.transform(X_test)
for i in range(2):
    print() 
print("Original Training Data:", X_train,end="\n")
for i in range(2):
    print() 
print("Scaled fit Training Data:", f"Mean: {X_train_scaled1.mean_}")
print(f"Variance: {X_train_scaled1.var_}")
for i in range(2):
    print() 
print("Scaled Transform Training Data:", X_train_scaled,end="")
for i in range(2):
    print() 
print("Original Test Data:", X_test,end="")
for i in range(2):
    print() 
print("Scaled Transform Test Data:", X_test_scaled,end="\n\n")
for i in range(2):
    print() 


var1=np.var(X_train[0])
print(type(X_train))
print((X_train))
me= [sublist[0] for sublist in X_train]
print(me)
m0=np.mean(me)
print(m0)
v0=np.var(me)
print (v0)


me1= [sublist[1] for sublist in X_train]
print(me1)
m1=np.mean(me1)
print(m1)
v1=np.var(me1)
print (v1)

