from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import os
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image


app = Flask(__name__)

# Load model
model_path = os.path.join("model", "model.pkl")
model = pickle.load(open(model_path, 'rb'))

# Load room tidiness image classification model
image_model_path = os.path.join("model", "room_tidiness_model.keras")
image_model = load_model(image_model_path)
image_labels = ['Clean', 'Messy']  # update if needed

app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/get_quote')
def get_quote():
    return render_template('get_quote.html')

@app.route('/quote_options', methods=['POST'])
def quote_options():
    cleaning_type = request.form['cleaning_type']
    return render_template('quote_options.html', cleaning_type=cleaning_type)

@app.route('/quote_form', methods=['POST'])
def quote_form():
    house_type = request.form['house_type']
    cleaning_type = request.form['cleaning_type']
    return render_template('quote_form.html', house_type=house_type, cleaning_type=cleaning_type)

@app.route('/predict_price', methods=['POST'])
def predict_price():
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    cleaning_type = request.form['cleaning_type']
    house_type = request.form['house_type']

    # Convert inputs into features for prediction
    input_features = [bedrooms, bathrooms,
                      1 if cleaning_type.lower() == 'deep' else 0,
                      1 if house_type.lower() == 'detached' else 0]

    work_hours = model.predict([input_features])[0]

    prices = {
        "economy": round(work_hours * 27, 2),
        "standard": round(work_hours * 30, 2),
        "premium": round(work_hours * 37, 2)
    }

    return render_template('quote_result.html', work_hours=round(work_hours, 2), prices=prices)

@app.route('/room_analysis', methods=['GET', 'POST'])
def room_analysis():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = image_model.predict(img_array)[0][0]
            label = "Messy" if prediction > 0.5 else "Clean"

            return render_template(
                'room_analysis.html',
                prediction=label,
                image_url=url_for('static', filename='uploads/' + filename)
            )

    return render_template('room_analysis.html')



@app.route('/analyze_room', methods=['POST'])
def analyze_room():
    if 'room_image' not in request.files:
        return render_template('room_analysis.html', prediction="No file uploaded")

    file = request.files['room_image']
    if file.filename == '':
        return render_template('room_analysis.html', prediction="No selected file")

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and preprocess the image
        img = load_img(filepath, target_size=(224, 224))  # adjust size to your model
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)

        prediction = image_model.predict(img_array)[0][0]
        predicted_label = "Messy" if prediction > 0.5 else "Clean"


        return render_template('room_analysis.html', prediction=predicted_label)

    return render_template('room_analysis.html', prediction="Error during upload")


@app.route('/insights')
def insights():
    return render_template('insights.html')

@app.route('/team')
def team():
    return render_template('team.html')

if __name__ == '__main__':
    app.run(debug=True)
