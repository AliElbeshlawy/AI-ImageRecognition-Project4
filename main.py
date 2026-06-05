import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load pre-trained model
model = MobileNetV2(weights='imagenet')

def predict_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    # Preprocess
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    predictions = model.predict(img_array)

    # Decode results
    results = decode_predictions(predictions, top=3)[0]

    print("\n🔍 Prediction Results:")
    for result in results:
        print(f"{result[1]}: {round(result[2]*100, 2)}%")

# Test with sample image
if __name__ == "__main__":
    image_path = "images/test.jpg" 
    predict_image(image_path)
