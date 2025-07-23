# Brain Tumor Classification using CNNs (PyTorch)
## Project Overview 🧠
This project implements a Convolutional Neural Network (CNN) for the classification of brain tumors from MRI images. It follows a robust modular coding pattern and leverages Data Version Control (DVC) for reproducibility, ensuring that the entire machine learning pipeline, from data ingestion to model deployment, is traceable and re-executable. The trained model is then deployed via a lightweight Flask web application, allowing for easy inference through a user-friendly interface.

**Achieved Performance:** The trained model achieved a remarkable 99.24% Test Accuracy on the unseen test dataset.

**Features** ✨
Modular Codebase: Organized into distinct components (data ingestion, transformation, model preparation, training, evaluation, prediction) for maintainability and scalability.

**Data Version Control (DVC):** Utilizes DVC to version datasets and model artifacts, ensuring reproducibility of experiments and traceability of results.

**PyTorch Framework:** Built using PyTorch for flexible and efficient deep learning model development.

**Transfer Learning:** Employs a pre-trained ResNet18 model, fine-tuned for brain tumor classification.

**Automated Pipeline:** An end-to-end ML pipeline orchestrated via main.py and defined in dvc.yaml.

**Comprehensive Evaluation:** Includes detailed evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix).

**Web Deployment (Flask):** A simple web application to upload MRI images and get real-time brain tumor predictions.

**Logging:** Integrated logging for monitoring pipeline execution and debugging.

## Project Structure 📁
The project adheres to a standard modular structure to keep components organized:
![Project Structure](project_structure.png)
 

## Setup & Installation 🛠️
Follow these steps to set up the project locally:

**Clone the repository:**

**git clone https://github.com/your-username/your-repo-name.git**
cd your-repo-name

(Replace your-username and your-repo-name with your actual GitHub details)

**Create a Conda environment (recommended):**

conda create -n brain_tumour_env python=3.9 -y
conda activate brain_tumour_env

Install Python dependencies:

pip install -r requirements.txt

**Note: This project uses PyTorch. Depending on your system and GPU availability, you might need to install PyTorch separately. For CUDA 11.8, use:**

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

For CPU-only, use:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Refer to the official PyTorch installation guide for more options.

Initialize DVC:

dvc init

Pull DVC-tracked data and models:
This command will download the dataset and pre-trained model artifacts that are versioned with DVC.

dvc pull

## How to Run the Project ▶️
There are two main ways to run this project: running the full ML pipeline and running the Flask web application.

**A. Running the ML Pipeline (Training & Evaluation)**
This will execute the entire machine learning pipeline from data ingestion to model evaluation.

Ensure your Conda environment is activated:

conda activate brain_tumour_env

Execute the main pipeline script:

python main.py

This command will sequentially run the following stages:

Stage 1: Data Ingestion: Downloads and unzips the raw dataset.

Stage 2: Data Transformation: Prepares data loaders with appropriate augmentations and transformations.

Stage 3: Prepare Base Model: Loads a pre-trained ResNet18 model, freezes its layers, and modifies the classification head.

Stage 4: Model Training: Trains the model on the prepared data, including early stopping and saving the best model.

Stage 5: Model Evaluation: Evaluates the trained model on the test set and saves detailed metrics.

You will see detailed logs in your terminal and in the logs/running_logs.log file. Upon successful completion, you will find the trained model in artifacts/model_training/model.pth and evaluation metrics in artifacts/model_evaluation/evaluation_metrics.json.

**B. Running the Flask Web Application (Deployment)**
This will start a local web server where you can upload images for prediction.

Ensure your Conda environment is activated:

conda activate brain_tumour_env

Execute the Flask application script:

python app.py

The terminal will show output indicating that the Flask server is running, typically on http://0.0.0.0:8080 or http://127.0.0.1:8080.

Access the web interface:
Open your web browser and navigate to the address provided (e.g., http://localhost:8080).
You can then use the interface to upload an MRI image, and the application will display the predicted brain tumor type along with confidence probabilities.

## Configuration ⚙️
config/config.yaml: This file contains all the static paths for artifacts and data, as well as configurations for each pipeline stage. Modify these paths if your local setup differs.

params.yaml: This file stores hyperparameters for training (e.g., IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS, PATIENCE). Adjust these values to experiment with different training settings.

## Results 📊
The model achieved impressive results during training and evaluation:

Best Validation Accuracy: 99.13%

## Final Test Accuracy: 99.24%

Detailed classification reports and confusion matrices are saved in artifacts/model_evaluation/evaluation_metrics.json after the model_evaluation stage.
## Input/Output Sample
Input image-![input](input.jpg)
Output image-![output](output.jpg)

## Future Enhancements (Next Steps) 💡
To take this project to the next level, consider exploring:

Advanced Model Architectures: Experiment with other state-of-the-art CNNs (e.g., ResNet50, EfficientNet) and advanced fine-tuning strategies.

Hyperparameter Optimization: Implement automated hyperparameter tuning using tools like Optuna or Weights & Biases Sweeps.

Comprehensive Experiment Tracking: Integrate with MLflow or Weights & Biases for more robust experiment logging and comparison.

CI/CD for ML: Set up automated workflows (e.g., GitHub Actions) to trigger retraining and re-evaluation on code or data changes.

Containerization (Docker): Dockerize the Flask application for consistent and portable deployment across environments.

Cloud Deployment: Deploy the Flask application to cloud platforms (AWS, GCP, Azure) for scalability and accessibility.

Model Monitoring: Implement tools to monitor model performance in production for data drift, concept drift, and prediction quality.

Improved Frontend: Enhance the web interface with a more sophisticated frontend framework (e.g., React, Vue.js).

## License 📄
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact ✉️
For any questions or suggestions, feel free to reach out:

[AMAN RAJ] - [amanraj07331@gmail.com]

Project Link:
