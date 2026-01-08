import joblib

print("🚀 Starting Training Pipeline...")

# --- 1. Load Data ---
try:
    df = pd.read_csv("Residential-Building-Data-Set.csv", header=1)
except FileNotFoundError:
    print("❌ Error: CSV file not found. Make sure it's in the folder.")
    exit()

# Rename for clarity
column_renames = {
    'V-1': 'Project_Locality',
    'V-2': 'Total_Floor_Area',
    'V-3': 'Lot_Area',
    'V-4': 'Total_Prelim_Est',
    'V-5': 'Prelim_Est_Unit_Cost',
    'V-6': 'Inflation_Index',
    'V-7': 'Duration',
    'V-8': 'Unit_Price_Start',
    'V-10': 'Actual_Cost'
}
df = df.rename(columns=column_renames)

# Drop Leakage
X = df.drop(columns=['V-9', 'Actual_Cost'])
y = df['Actual_Cost']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Synthetic Data Generation ---
print("🧪 Augmenting data with Synthetic samples...")
train_data_combined = pd.concat([X_train, y_train], axis=1)
scaler_gen = StandardScaler()
train_data_scaled = scaler_gen.fit_transform(train_data_combined)

kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(train_data_scaled)

# Generate 300 new samples
synthetic_scaled = kde.sample(300, random_state=42)
synthetic_data = scaler_gen.inverse_transform(synthetic_scaled)
synthetic_df = pd.DataFrame(synthetic_data, columns=train_data_combined.columns)

X_synthetic = synthetic_df.drop(columns=['Actual_Cost'])
y_synthetic = synthetic_df['Actual_Cost']

# Combine
X_final = pd.concat([X_train, X_synthetic], axis=0)
y_final = pd.concat([y_train, y_synthetic], axis=0)

# --- 3. Modeling ---
print("🧠 Training Ridge Regression...")
scaler_model = StandardScaler()
X_final_scaled = scaler_model.fit_transform(X_final)

ridge = RidgeCV(cv=5)
ridge.fit(X_final_scaled, y_final)

score = ridge.score(scaler_model.transform(X_test), y_test)
print(f"✅ Model Trained. Test R2 Score: {score:.4f}")

# --- 4. Save Artifacts ---
artifacts = {
    'model': ridge,
    'scaler': scaler_model,
    'default_values': X_train.mean(),
    'feature_names': X_train.columns.tolist()
}

joblib.dump(artifacts, 'construction_model.joblib')
print("💾 Model saved to 'construction_model.joblib'")
