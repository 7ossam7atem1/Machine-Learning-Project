import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QLabel,
    QFileDialog,
    QComboBox,
    QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from regression import TrainingWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Project")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        # Define left and right widgets
        self.left_widget = QWidget()
        self.right_widget = QWidget()
        self.visualize_widget = QWidget()

        # Define layouts for left, right, and visualize widgets
        self.left_layout = QVBoxLayout(self.left_widget)
        self.right_layout = QVBoxLayout(self.right_widget)
        self.visualize_layout = QVBoxLayout(self.visualize_widget)

        # Add left, right, and visualize widgets to main layout
        self.layout.addWidget(self.left_widget)
        self.layout.addWidget(self.right_widget)
        self.layout.addWidget(self.visualize_widget)

        # Add preprocess button to left layout
        self.preprocess_button = QPushButton("Preprocess Data")
        self.preprocess_button.clicked.connect(self.preprocess_data)
        self.left_layout.addWidget(self.preprocess_button)

        # Add upload file button to left layout
        self.upload_button = QPushButton("Upload Your File")
        self.upload_button.clicked.connect(self.load_file)
        self.left_layout.addWidget(self.upload_button)

        # Add visualize button to left layout
        # self.visualize_button = QPushButton("Visualization")
        # self.visualize_button.clicked.connect(self.show_visualize_panel)
        # self.left_layout.addWidget(self.visualize_button)

        # Add table view to right layout
        self.table_view = QTableWidget()
        self.right_layout.addWidget(self.table_view)

        # Add data description table view to right layout
        self.data_desc_table_view = QTableWidget()
        self.right_layout.addWidget(self.data_desc_table_view)

        # Initialize data variable
        self.raw_data = None
        self.preprocessed_data = None

        # Set up visualize panel
        # self.visualize_label = QLabel("Visualize Panel")
        # self.visualize_label.setAlignment(Qt.AlignCenter)
        #self.visualize_layout.addWidget(self.visualize_label)

        self.visualize_type_combo_box = QComboBox()
        self.visualize_type_combo_box.addItem("Line Plot")
        self.visualize_type_combo_box.addItem("Scatter Plot")
        self.visualize_type_combo_box.addItem("Box Plot")
        self.visualize_layout.addWidget(self.visualize_type_combo_box)

        self.visualize_button = QPushButton("Visualize Data")
        self.visualize_button.clicked.connect(self.visualize_data)
        self.visualize_layout.addWidget(self.visualize_button)

        self.train_button = QPushButton("Regression")
        self.train_button.clicked.connect(self.show_training_window)
        self.visualize_layout.addWidget(self.train_button)

        self.perform_classification_button = QPushButton("Perform Classification")
        self.perform_classification_button.clicked.connect(self.perform_classification)
        self.visualize_layout.addWidget(self.perform_classification_button)

        self.visualize_image_label = QLabel()
        self.visualize_layout.addWidget(self.visualize_image_label)

    def perform_classification(self):
        if self.raw_data is None:
            print("No data to classify")
            return

        # Perform classification using SVM and KNN
        svm_results = self.perform_svm_classification(self.raw_data)
        knn_results = self.perform_knn_classification(self.raw_data)

        # Display results in a pop-up window
        self.display_classification_results(svm_results, knn_results)

    def perform_svm_classification(self, data):
        # Perform SVM classification on preprocessed data
        # Here you need to provide the necessary code to perform SVM classification
        # You can use the preprocessed data stored in self.preprocessed_data


        y = data['Outcome']
        X = data.drop('Outcome', axis=1)

        svm_model = SVC()
        svm_model.fit(X, y)
        y_pred = svm_model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        svm_results = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        return svm_results

    def perform_knn_classification(self, data):
        # Perform KNN classification on preprocessed data
        # Here you need to provide the necessary code to perform KNN classification
        # You can use the preprocessed data stored in self.preprocessed_data

        # Placeholder code
        y = data['Outcome']
        X = data.drop('Outcome', axis=1)

        knn_model = KNeighborsClassifier()
        knn_model.fit(X, y)
        y_pred = knn_model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        knn_results = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        return knn_results

    def display_classification_results(self, svm_results, knn_results):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Classification Results")
        msg_box.setText(f"SVM Results:\nAccuracy: {svm_results['Accuracy']}\n"
                        f"Precision: {svm_results['Precision']}\n"
                        f"Recall: {svm_results['Recall']}\n"
                        f"F1 Score: {svm_results['F1 Score']}\n\n"
                        f"KNN Results:\nAccuracy: {knn_results['Accuracy']}\n"
                        f"Precision: {knn_results['Precision']}\n"
                        f"Recall: {knn_results['Recall']}\n"
                        f"F1 Score: {knn_results['F1 Score']}")
        msg_box.exec()

    def load_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File", "", "CSV Files (*.csv)", options=options)

        if file_name:
            self.raw_data = pd.read_csv(file_name)
            self.update_table_view(self.raw_data)

    def update_table_view(self, data):
        self.table_view.setRowCount(data.shape[0])
        self.table_view.setColumnCount(data.shape[1])
        self.table_view.setHorizontalHeaderLabels(data.columns)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.table_view.setItem(i, j, QTableWidgetItem(str(data.iloc[i, j])))

    # def update_data_desc_table_view(self, data):
    #     self.data_desc_table_view.setRowCount(7)
    #     self.data_desc_table_view.setColumnCount(2)
    #     self.data_desc_table_view.setHorizontalHeaderLabels(['Statistic', 'Value'])
    #
    #     self.data_desc_table_view.setItem(0, 0, QTableWidgetItem("Number of Rows"))
    #     self.data_desc_table_view.setItem(0, 1, QTableWidgetItem(str(data.shape[0])))
    #
    #     self.data_desc_table_view.setItem(1, 0, QTableWidgetItem("Number of Columns"))
    #     self.data_desc_table_view.setItem(1, 1, QTableWidgetItem(str(data.shape[1])))
    #
    #     self.data_desc_table_view.setItem(2, 0, QTableWidgetItem("Column Names"))
    #     self.data_desc_table_view.setItem(2, 1, QTableWidgetItem(", ".join(data.columns)))
    #
    #     self.data_desc_table_view.setItem(3, 0, QTableWidgetItem("Data Types"))
    #     self.data_desc_table_view.setItem(3, 1, QTableWidgetItem(", ".join([str(data[col].dtype) for col in data.columns])))
    #
    #     self.data_desc_table_view.setItem(4, 0, QTableWidgetItem("Missing Values"))
    #     self.data_desc_table_view.setItem(4, 1, QTableWidgetItem(str(data.isnull().sum().sum())))
    #
    #     self.data_desc_table_view.setItem(5, 0, QTableWidgetItem("Unique Values"))
    #     self.data_desc_table_view.setItem(5, 1, QTableWidgetItem(str(data.nunique().sum())))
    #
    #     self.data_desc_table_view.setItem(6, 0, QTableWidgetItem("Summary Statistics"))
    #     self.data_desc_table_view.setItem(6, 1, QTableWidgetItem(str(data.describe())))
    def update_data_desc_table_view(self, data):
        self.data_desc_table_view.setRowCount(13)
        self.data_desc_table_view.setColumnCount(2)
        self.data_desc_table_view.setHorizontalHeaderLabels(['Statistic', 'Value'])

        self.data_desc_table_view.setItem(0, 0, QTableWidgetItem("Number of Rows"))
        self.data_desc_table_view.setItem(0, 1, QTableWidgetItem(str(data.shape[0])))

        self.data_desc_table_view.setItem(1, 0, QTableWidgetItem("Number of Columns"))
        self.data_desc_table_view.setItem(1, 1, QTableWidgetItem(str(data.shape[1])))

        self.data_desc_table_view.setItem(2, 0, QTableWidgetItem("Column Names"))
        self.data_desc_table_view.setItem(2, 1, QTableWidgetItem(", ".join(data.columns)))

        self.data_desc_table_view.setItem(3, 0, QTableWidgetItem("Data Types"))
        self.data_desc_table_view.setItem(3, 1,
                                          QTableWidgetItem(", ".join([str(data[col].dtype) for col in data.columns])))

        self.data_desc_table_view.setItem(4, 0, QTableWidgetItem("Missing Values"))
        self.data_desc_table_view.setItem(4, 1, QTableWidgetItem(str(data.isnull().sum().sum())))

        self.data_desc_table_view.setItem(5, 0, QTableWidgetItem("Unique Values"))
        self.data_desc_table_view.setItem(5, 1, QTableWidgetItem(str(data.nunique().sum())))

        self.data_desc_table_view.setItem(6, 0, QTableWidgetItem("Minimum"))
        self.data_desc_table_view.setItem(6, 1,
                                          QTableWidgetItem(", ".join([str(data[col].min()) for col in data.columns])))

        self.data_desc_table_view.setItem(7, 0, QTableWidgetItem("Maximum"))
        self.data_desc_table_view.setItem(7, 1,
                                          QTableWidgetItem(", ".join([str(data[col].max()) for col in data.columns])))

        self.data_desc_table_view.setItem(8, 0, QTableWidgetItem("Standard Deviation"))
        self.data_desc_table_view.setItem(8, 1,
                                          QTableWidgetItem(", ".join([str(data[col].std()) for col in data.columns])))

        self.data_desc_table_view.setItem(9, 0, QTableWidgetItem("25th Percentile"))
        self.data_desc_table_view.setItem(9, 1, QTableWidgetItem(
            ", ".join([str(data[col].quantile(0.25)) for col in data.columns])))

        self.data_desc_table_view.setItem(10, 0, QTableWidgetItem("50th Percentile (Median)"))
        self.data_desc_table_view.setItem(10, 1, QTableWidgetItem(
            ", ".join([str(data[col].quantile(0.5)) for col in data.columns])))

        self.data_desc_table_view.setItem(11, 0, QTableWidgetItem("75th Percentile"))
        self.data_desc_table_view.setItem(11, 1, QTableWidgetItem(
            ", ".join([str(data[col].quantile(0.75)) for col in data.columns])))

        # self.data_desc_table_view.setItem(12, 0, QTableWidgetItem("Summary Statistics"))
        # self.data_desc_table_view.setItem(12, 1, QTableWidgetItem(""))

    def preprocess_data(self):
        if self.raw_data is not None:
            self.preprocessed_data = self.raw_data.copy()

            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            self.preprocessed_data = pd.DataFrame(imputer.fit_transform(self.preprocessed_data), columns=self.preprocessed_data.columns)

            # Dimensionality reduction with PCA
            pca = PCA(n_components=9)
            principal_components = pca.fit_transform(self.preprocessed_data)
            #PCA COLUMNS ADDED MANUALLY:
            # self.preprocessed_data = pd.DataFrame(principal_components, columns=['PC1', 'PC2' , 'PC3' ])
            self.preprocessed_data = pd.DataFrame(principal_components, columns=self.preprocessed_data.columns[:len(principal_components[0])])

            # Normalize data using StandardScaler and MinMaxScaler
            numeric_cols = self.preprocessed_data.select_dtypes(include=['float64', 'int64']).columns
            standard_scaler = StandardScaler()
            minmax_scaler = MinMaxScaler()
            self.preprocessed_data[numeric_cols] = standard_scaler.fit_transform(self.preprocessed_data[numeric_cols])
            self.preprocessed_data[numeric_cols] = minmax_scaler.fit_transform(self.preprocessed_data[numeric_cols])

            # Encode categorical columns using LabelEncoder and OneHotEncoder
            categorical_cols = self.preprocessed_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if len(self.preprocessed_data[col].unique()) == 2:
                    encoder = LabelEncoder()
                    self.preprocessed_data[col] = encoder.fit_transform(self.preprocessed_data[col])
                else:
                    encoder = OneHotEncoder()
                    encoded_data = encoder.fit_transform(self.preprocessed_data[[col]])
                    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=[f"{col}_{val}" for val in encoder.categories_[0]])
                    self.preprocessed_data = pd.concat([self.preprocessed_data, encoded_df], axis=1)
                    self.preprocessed_data = self.preprocessed_data.drop(columns=[col])

            # Sample data if it is unbalanced
            target_col = self.preprocessed_data.columns[-1]
            if len(self.preprocessed_data[target_col].unique()) == 2:
                over_sampler = RandomOverSampler()
                under_sampler = RandomUnderSampler()
                X = self.preprocessed_data.drop(columns=[target_col])
                y = self.preprocessed_data[target_col]
                X_resampled, y_resampled = over_sampler.fit_resample(X, y)
                X_resampled, y_resampled = under_sampler.fit_resample(X_resampled, y_resampled)
                self.preprocessed_data = pd.concat([X_resampled, y_resampled], axis=1)

            self.update_table_view(self.preprocessed_data)
            self.update_data_desc_table_view(self.preprocessed_data)
        else:
            print("No data to preprocess.")

    def show_visualize_panel(self):
        if self.preprocessed_data is not None:
            self.visualize_widget.show()
        else:
            print("No preprocessed data to visualize.")

    def visualize_data(self):
        if self.preprocessed_data is not None:
            plt.clf()
            if self.visualize_type_combo_box.currentIndex() == 0:
                plt.plot(self.preprocessed_data.iloc[:, 0], self.preprocessed_data.iloc[:, 1])
            elif self.visualize_type_combo_box.currentIndex() == 1:
                plt.scatter(self.preprocessed_data.iloc[:, 0], self.preprocessed_data.iloc[:, 1])
            elif self.visualize_type_combo_box.currentIndex() == 2:
                plt.boxplot(self.preprocessed_data)
            plt.savefig('plot.png')
            image = QImage('plot.png')
            self.visualize_image_label.setPixmap(QPixmap.fromImage(image))
        else:
            print("No preprocessed data to visualize.")

    def show_training_window(self):
        # if self.preprocessed_data is None:
        #     print("No preprocessed data to train.")
        #     return
        self.training_window = TrainingWindow()
        self.training_window.setWindowTitle("Training Window")
        self.training_window.show()
        print(self.training_window)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())