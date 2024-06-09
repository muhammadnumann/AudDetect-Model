from fastapi import FastAPI, UploadFile
import shutil
import os
import numpy as np
from librosa import load, feature
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()


async def predict_deepfake(audio_file_path, max_length=500, threshold=0.5):

  model_path = os.getenv("MODEL_PATH")
  # Error handling
  if not os.path.exists(audio_file_path):
    raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

  # Load audio
  try:
    audio, _ = load(audio_file_path, sr=16000)
  except Exception as e:
    print(f"Error loading audio file: {e}")
    return None

  # Extract MFCC features
  mfccs = feature.mfcc(y=audio, sr=16000, n_mfcc=50)

  if mfccs.shape[1] < max_length:
    mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
  else:
    mfccs = mfccs[:, :max_length]


  mfccs = mfccs.reshape((1, mfccs.shape[0], mfccs.shape[1], 1))

  try:
    model = load_model(model_path)
  except Exception as e:
    print(f"Error loading model: {e}")
    return None


  prediction = model.predict(mfccs)
  predicted_label = "real" if prediction[0][0] < threshold else "fake"

  return predicted_label


app = FastAPI()

@app.post("/")
async def upload_file(file: UploadFile = UploadFile(...)):
    try:
        current_path = os.getcwd()
      
        with open(os.path.join(current_path, file.filename) , "wb") as f:
            shutil.copyfileobj(file.file, f)        

        audio_file = file.filename
        prediction = await predict_deepfake(audio_file)
        print("Predicted label: ",prediction)

        try:
            os.remove(audio_file)
        except OSError as e:
            print(f"Error deleting file {audio_file}: {e}")
            
        return {"message": "File saved successfully","isReal":prediction=='real'}

    except Exception as e:
        print(f"Error saving file '{file.filename}': {e}")
        return {"error": str(e)}