import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    # ✅ Fix deprecation warning for fillna
    df.ffill(inplace=True)

    # 🔠 Encode categorical variables
    label_encoders = {}
    for column in ['Gender', 'Subscription Status', 'Discount Applied', 'Product Category']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # 🎯 Split features and target
    X = df.drop('Product Category', axis=1)
    y = df['Product Category']

    # 📏 Normalize numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ⚖️ Balance dataset using SMOTE with safe k_neighbors
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, scaler, label_encoders
