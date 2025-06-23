from flask import Flask, render_template, request, send_file, session
import pandas as pd
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from flask_session import Session

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    session.clear()  # Reset session every new visit to home
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    result = {}
    if request.method == 'POST':
        file = request.files['file']
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            df = pd.read_csv(path)

            target_col = request.form['target']
            model_type = request.form['model_type']
            save_model = request.form.get('save_model')

            X = df.drop(columns=[target_col])
            y = df[target_col]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            if model_type == 'classification':
                model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                result['score'] = f"Accuracy: {score*100 :.2f} %"
                result['cm'] = confusion_matrix(y_test, y_pred).tolist()

                if save_model == 'yes':
                    model.save_model('model_classification.json')
                    result['model_saved'] = 'classification'
                    session['model_file'] = 'model_classification.json'

            elif model_type == 'regression':
                model = xgb.XGBRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                result['score'] = f"MSE: {mse:.2f}, R2: {r2:.2f}"

                if save_model == 'yes':
                    model.save_model('model_regression.json')
                    result['model_saved'] = 'regression'
                    session['model_file'] = 'model_regression.json'

            # Only allow predict after train in this session
            session['allow_predict'] = True
            session['model_type'] = model_type

            return render_template('train.html', result=result)
    return render_template('train.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not session.get('allow_predict'):
        return "❌ You must train a model first in this session!", 403

    predictions = None
    if request.method == 'POST':
        predict_file = request.files['predict_file']
        model_type = session.get('model_type')  # Use session's model type
        if predict_file:
            path = os.path.join(UPLOAD_FOLDER, predict_file.filename)
            predict_file.save(path)
            new_data = pd.read_csv(path)

            scaler = pickle.load(open('scaler.pkl', 'rb'))
            new_data_scaled = scaler.transform(new_data)

            if model_type == 'classification':
                model = xgb.XGBClassifier()
                model.load_model('model_classification.json')
            elif model_type == 'regression':
                model = xgb.XGBRegressor()
                model.load_model('model_regression.json')

            preds = model.predict(new_data_scaled)
            new_data['Prediction'] = preds
            predictions = new_data.head().to_html(classes='table table-striped', index=False)
            new_data.to_csv('prediction_results.csv', index=False)
    return render_template('predict.html', predictions=predictions)

@app.route('/download_model/<model_type>')
def download_model(model_type):
    if model_type == 'classification':
        return send_file('model_classification.json', as_attachment=True)
    elif model_type == 'regression':
        return send_file('model_regression.json', as_attachment=True)

@app.route('/download_predictions')
def download_predictions():
    return send_file('prediction_results.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
