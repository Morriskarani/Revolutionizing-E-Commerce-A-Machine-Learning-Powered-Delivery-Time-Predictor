from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import plotly.express as px

# Initialize Flask app
app = Flask(__name__)

# --- Load Dataset for Training ---
df = pd.read_csv("Reduced_E_Commerce original.csv")

# Encode Categorical Columns
label_encoders = {}
for col in ["Warehouse_block", "Mode_of_Shipment"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Reached.on.Time_Y.N", axis=1)
y = df["Reached.on.Time_Y.N"]

# --- Check for Existing Model or Train + Save ---
model_path = "decision_tree_model.pkl"

if os.path.exists(model_path):
    best_dt = joblib.load(model_path)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 4, 6],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train_smote, y_train_smote)

    best_dt = grid.best_estimator_
    joblib.dump(best_dt, model_path)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/visualizations", methods=["GET"])
def visualizations():
    df = pd.read_csv("E_Commerce.csv")
    shipment = request.args.get("shipment")
    discount = request.args.get("discount", type=int)
    importance = request.args.getlist("importance")

    if shipment:
        df = df[df['Mode_of_Shipment'] == shipment]
    if discount is not None:
        df = df[df['Discount_offered'] <= discount]
    if importance:
        df = df[df['Product_importance'].isin(importance)]

    numeric_df = df.select_dtypes(include=[float, int])

    fig1 = px.histogram(df, x="Cost_of_the_Product", title="Cost Distribution")
    fig2 = px.pie(df, names="Product_importance", title="Product Importance Share")
    fig3 = px.scatter(df, x="Cost_of_the_Product", y="Weight_in_gms", color="Mode_of_Shipment", title="Cost vs Weight")
    fig4 = px.line(df, x="Discount_offered", y="Customer_rating", title="Discount vs Customer Rating")
    fig5 = px.imshow(numeric_df.corr(), title="Correlation Heatmap")

    return render_template("visualizations.html", 
                           chart_html1=fig1.to_html(full_html=False), 
                           chart_html2=fig2.to_html(full_html=False), 
                           chart_html3=fig3.to_html(full_html=False),
                           chart_html4=fig4.to_html(full_html=False), 
                           chart_html5=fig5.to_html(full_html=False))

@app.route("/dataset", methods=["GET", "POST"])
def dataset():
    df = pd.read_csv("E_Commerce.csv")
    search = ""

    if request.method == "POST":
        search = request.form.get("search", "").lower()
        if search:
            df = df[df.apply(lambda row: row.astype(str).str.lower().str.contains(search).any(), axis=1)]

    data_types = df.dtypes.reset_index()
    data_types.columns = ['Column', 'Data Type']

    return render_template("dataset.html",
                           table=df.to_html(classes="table table-striped", index=False),
                           data_types=data_types.to_html(classes="table table-bordered", index=False),
                           search_query=search)

@app.route('/download')
def download():
    return send_file('E_Commerce.csv', mimetype='text/csv', download_name='project_dataset.csv', as_attachment=True)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs
        warehouse = request.form.get("warehouse_block")
        shipment = request.form.get("shipment_mode")
        customer_calls = request.form.get("customer_calls")
        rating = request.form.get("customer_rating")
        prior = request.form.get("prior_purchases")
        discount = request.form.get("discount_offered")
        weight = request.form.get("weight_in_gms")

        # Validate inputs
        if not all([warehouse, shipment, customer_calls, rating, prior, discount, weight]):
            return jsonify({'error': 'All fields are required.'}), 400

        # Convert numeric inputs to integers
        try:
            customer_calls = int(customer_calls)
            rating = int(rating)
            prior = int(prior)
            discount = int(discount)
            weight = int(weight)
        except ValueError:
            return jsonify({'error': 'Numeric fields must be valid integers.'}), 400

        # Validate rating range
        if not (1 <= rating <= 5):
            return jsonify({'error': 'Customer rating must be between 1 and 5.'}), 400

        # Validate non-negative values
        if any(x < 0 for x in [customer_calls, prior, discount, weight]):
            return jsonify({'error': 'Numeric fields cannot be negative.'}), 400

        # Validate categorical inputs
        valid_warehouses = list(label_encoders["Warehouse_block"].classes_)
        valid_shipments = list(label_encoders["Mode_of_Shipment"].classes_)
        if warehouse not in valid_warehouses:
            return jsonify({'error': f'Invalid warehouse block. Must be one of {valid_warehouses}.'}), 400
        if shipment not in valid_shipments:
            return jsonify({'error': f'Invalid shipment mode. Must be one of {valid_shipments}.'}), 400

        # Encode categorical inputs
        warehouse = label_encoders["Warehouse_block"].transform([warehouse])[0]
        shipment = label_encoders["Mode_of_Shipment"].transform([shipment])[0]

        # Prepare the input DataFrame
        new_order = pd.DataFrame([[warehouse, shipment, customer_calls, rating, prior, discount, weight]], columns=X.columns)

        # Predict using the trained model
        prediction = best_dt.predict(new_order)

        # Prepare reasoning based on feature importance
        feature_importances = pd.Series(best_dt.feature_importances_, index=X.columns)
        top_features = feature_importances.sort_values(ascending=False).head(3)

        new_values = new_order.iloc[0]
        reasons = []

        for feature in top_features.index:
            value = new_values[feature]
            if feature == "Customer_rating" and value >= 4:
                reasons.append("✅ High customer rating, likely better experience.")
            elif feature == "Discount_offered" and value > 20:
                reasons.append("✅ Generous discount offered, could prioritize order.")
            elif feature == "Weight_in_gms" and value < 3000:
                reasons.append("✅ Lightweight package, easier to deliver fast.")
            elif feature == "Weight_in_gms" and value >= 5000:
                reasons.append("⚠️ Heavy package, may cause delay.")
            elif feature == "Customer_rating" and value <= 2:
                reasons.append("⚠️ Low customer rating, possible service issue.")

        prediction_result = "On Time" if prediction[0] == 1 else "Late"
        return jsonify({
            'prediction': prediction_result,
            'reasons': reasons
        })

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)




