import pandas as pd
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QSlider
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics
from design import Ui_Dialog

class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()

        #Connecting the .ui with .py scripts
        self.df = None
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.UploadFile.clicked.connect(self.buttonClick)
        self.ui.testSize.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ui.testSize.setTickInterval(1)
        self.ui.testSize.valueChanged.connect(self.tsize)
        self.ui.alphaSize.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ui.alphaSize.setTickInterval(1)
        self.ui.alphaSize.valueChanged.connect(self.Asize)
        self.ui.MAE.clicked.connect(self.showtrain)
        self.ui.MSE.clicked.connect(self.showtrain)
        self.ui.RMSE.clicked.connect(self.showtrain)
        self.ui.train.clicked.connect(self.TRAIN)
        self.ui.test.clicked.connect(self.TEST)

    # Button clicked to display the csv file in the dilaog
    def buttonClick(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.df = pd.read_csv(file_path)
        self.x = self.df.iloc[:,:-1].values
        self.y = self.df.iloc[:,1].values
        self.ui.ShowData.setEnabled(True)
        self.ui.ShowData.setText(self.df.to_string())
        report = self.df.describe()
        print(report)
        self.ui.Datadesc.setText(report.to_string())
        self.ui.testSize.setEnabled(True)


#   SPLITTING THE DATA INTO TRAIN AND TEST
    def TRAIN(self):
        self.ui.train_error.setPlainText(str(self.error_train))
        self.ui.test.setEnabled(True)

    def TEST(self):
         self.ui.test_error.setPlainText(str(self.error_test))
         self.ui.test.setEnabled(True)

# Display the train if one of the radio buttons is checked
    def showtrain(self):
        self.ui.train.setEnabled(True)
        self.lasso = Lasso(alpha=self.alpha)
        self.lasso.fit(self.X_train, self.y_train)
        self.y_pred = self.lasso.predict(self.X_test)
        self.train_pred = self.lasso.predict(self.X_train)

        if self.ui.MAE.isChecked():
            self.error_train = metrics.mean_absolute_error(self.y_train,self.train_pred)
            self.error_test = metrics.mean_absolute_error(self.y_test,self.y_pred)

        elif self.ui.MSE:
            self.error_train = metrics.mean_squared_error(self.y_train,self.train_pred)
            self.error_test = metrics.mean_squared_error(self.y_test,self.y_pred)

        elif self.ui.RMSE:
            self.error_train = np.sqrt(metrics.mean_squared_error(self.y_train,self.train_pred))
            self.error_test = np.sqrt(metrics.mean_squared_error(self.y_train,self.train_pred))

    def showtest(self):
        self.ui.test.setEnabled(True)
        print(self.x)
        print(self.y)
        print(self.test_size)


     # is called when the valueChanged signal of the testSize slider is emitted.
    def tsize(self):
        self.test_size=self.sender().value()/100
        # self.y_test = train_test_split(self.x, self.y, test_size=self.test_size, random_state=0)
        self.ui.testLb.setText(str(self.test_size))
        self.ui.testLb.adjustSize() # Expands label size as numbers get larger
        self.ui.alphaSize.setEnabled(True)

    #is called when the valueChanged signal of the alphaSize slider is emitted.
    def Asize(self):
        self.alpha = self.sender().value()
        self.ui.alphaLb.setText(str(self.alpha/100))
        self.ui.alphaLb.adjustSize() # Expands label size as numbers get larger
        self.ui.MSE.setEnabled(True)
        self.ui.MAE.setEnabled(True)
        self.ui.RMSE.setEnabled(True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y,test_size=self.test_size,random_state=0)
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.y_train)
        self.y_pred = self.regressor.predict(self.X_test)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingWindow()
    window.show()
    sys.exit(app.exec())
