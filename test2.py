from sklearn.preprocessing import StandardScaler

# Sample data
X = [[1, 2], [2, 3], [3, 4]]
print(X)
# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler (learn mean and std from data)
X_train_scaled1=scaler.fit(X)
for i in range(2):
    print() 

print("Scaled fit Training Data:", f"Mean: {X_train_scaled1.mean_}")
print(f"Variance: {X_train_scaled1.var_}")
for i in range(2):
    print() 

# Apply transformation (standardization)
X_transformed = scaler.transform(X)
for i in range(2):
    print() 

print("Transformed_Data:",X_transformed)