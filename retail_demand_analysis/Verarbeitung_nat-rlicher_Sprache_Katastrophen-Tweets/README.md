# Natural Language Processing: Disaster Tweets Analysis

This project focuses on classifying whether tweets are about real disasters or not. It demonstrates advanced NLP techniques and a commitment to professional engineering standards.

## 🛠 Technical Standards (Engineering Judgment)
To ensure scalability and system stability, I follow a set of core principles in my development workflow:

* **Core Reserve Standard (1 Core Frei):** All ML computations are configured to leave at least one CPU core free to maintain system responsiveness and stability.
* **Memory Efficiency:** Strict RAM management is practiced by overwriting temporary DataFrames (`df_TEMP`) to minimize memory footprint during large-scale data processing.
* **Data Organization:** Priority is given to storing and loading data in **Packet-types** (structured packages) to ensure fast I/O and data integrity.
* **Progress Monitoring:** Long-running tasks, such as LSTM training or data cleaning, always include `tqdm` progress bars for transparent status tracking.


## 🚀 Project Overview
* **Goal:** Binary classification of disaster-related tweets.
* **Technologies:** Python, Pandas, Scikit-learn, TensorFlow/Keras (LSTM).
* **Status Tracking:** Verbose training logs and progress bars are standard.

## 📊 Data & Results
Numerical data and supporting tables for visualizations are included within the notebooks to ensure full transparency of the results.
