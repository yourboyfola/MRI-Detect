# 🧠 Brain Tumor MRI Classifier (MobileNetV2)

## 🚀 Overview
This project is a deep learning classifier deployed using **Streamlit** that analyzes brain MRI scans to classify them into four categories: **Glioma Tumor**, **Meningioma Tumor**, **Pituitary Tumor**, or **No Tumor**.

The model was built using Transfer Learning with the **MobileNetV2** architecture and achieved a validation accuracy of over 90%.

## ✨ Try the App Live!
**[https://mridetect.streamlit.app](https://mridetect.streamlit.app)**

## 💻 Technical Details

### Model Architecture
* **Base Model:** MobileNetV2 (pre-trained on ImageNet)
* **Approach:** Two-phase Transfer Learning (Feature Extraction followed by Fine-tuning the top 50 layers)
* **Classification:** 4-class Softmax layer
* **Training Framework:** TensorFlow/Keras
* **Accuracy:** >90% Validation Accuracy

### Deployment Stack
* **Frontend/Web Framework:** Streamlit
* **Deployment Platform:** Streamlit Community Cloud

## 🛠️ Local Setup and Installation

To run this application locally, clone the repository and install the dependencies:

1.  **Clone the repository:**
```bash
    git clone https://github.com/yourboyfola/MRI-Detect
    cd MRI-Detect
```

2.  **Install dependencies:**
```bash
    pip install -r requirements.txt
```

3.  **Run the Streamlit app:**
```bash
    streamlit run app.py
```

## ⚠️ Disclaimer
This is a demonstration model developed purely for educational purposes and to showcase machine learning deployment. **It is NOT intended for medical diagnosis.** Always consult qualified medical professionals for any health concerns or diagnostic decisions.

## 👤 Developer
**Folarin Eribake**
* **GitHub:** [yourboyfola](https://github.com/yourboyfola)
* **LinkedIn:** [Folarin Eribake](https://www.linkedin.com/in/folarin-eribake-80a63a281/)
