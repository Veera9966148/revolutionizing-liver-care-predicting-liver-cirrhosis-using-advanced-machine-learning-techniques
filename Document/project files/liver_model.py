# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load Dataset
df = pd.read_csv("cleaned_indian_liver_patient.csv")

# Step 3: Preprocessing
df.dropna(inplace=True)  # remove missing rows
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})  # convert gender

X = df.drop(['Dataset'], axis=1)  # input features
y = df['Dataset']                # target label

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Predict & Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model Accuracy:", accuracy)