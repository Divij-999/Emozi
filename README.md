
# 🎭 Emozi - Real-Time Facial Emotion Detection

**Emozi** is a deep learning-based Python project that detects human facial emotions in real-time using your webcam. It uses a Convolutional Neural Network (CNN) built with TensorFlow and trained on the FER-2013 dataset, along with OpenCV for live face detection.

---

## 🔧 Setup Instructions

### 1️⃣ Install Python 3.10.9

Download: [https://www.python.org/downloads/release/python-3109/](https://www.python.org/downloads/release/python-3109/)

Check version:

```bash
python --version
```

---

### 2️⃣ Create Virtual Environment

```bash
# Run CMD as Administrator
D:
cd "D:\user\emotion_project"
python -m venv emotion_env
emotion_env\Scripts\activate
```

---

### 3️⃣ Install Required Libraries

Create a `requirements.txt` file with:

```
tensorflow==2.15.0
opencv-python==4.8.1.78
numpy==1.24.4
matplotlib==3.7.1
pandas==1.5.3
scikit-learn==1.2.2
```

Then run:

```bash
pip install -r requirements.txt
```

If pip isn't recognized:

```bash
python -m ensurepip
python -m pip install --upgrade pip
```

---

### 4️⃣ Download Dataset

Get the FER-2013 dataset from Kaggle:  
[https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

Extract and place `fer2013.csv` into:

```
D:\user\emotion_project\archive\
```

---

### 5️⃣ Train the Model

Run:

```bash
python main.py
```

✔️ Trains the CNN  
💾 Saves model as `emotion_model.h5`  
📈 Displays accuracy graph

---

### 6️⃣ Run Live Prediction

Use webcam for real-time emotion detection:

```bash
python predict_live.py
```

Press `Q` to quit the window.

---

## 📁 Project Structure

```
emotion_project/
├── archive/               # Dataset folder
├── emotion_env/           # Virtual environment
├── main.py                # Model training
├── predict_live.py        # Webcam prediction
├── emotion_model.h5       # Trained model (after training)
├── requirements.txt       # Dependencies
```

---

## 👨‍💻 Coder

**Divij Modi**  
Passionate about AI, ML, and solving real-world problems.

---

## 🧪 License

This project is licensed under the **MIT License**.

---

Happy Emoting with **Emozi**! 😄
