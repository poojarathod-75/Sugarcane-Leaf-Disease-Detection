from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load YOLO model
model = YOLO("C:\\Users\\akash\\Desktop\\sugercane_leaf_backup\\weights\\best.pt")  # Make sure path is correct

# Create upload folder
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    try:
        results = model(img_path)
        output_path = os.path.join(OUTPUT_FOLDER, "output.jpg")
        results[0].save(filename=output_path)

        return jsonify({
            "status": "success",
            "output_image": "/" + output_path.replace("\\", "/")
        })

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)