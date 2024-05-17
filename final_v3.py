
##
##
## dashboard with eeg model and bluetooth connection


import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import bluetooth
import csv
import pickle
import time
import matplotlib.pyplot as plt

class EEGPredictionGUI:
    def __init__(self, master):
        self.master = master
        master.title("EEG Prediction GUI")
        master.geometry("750x750")

        self.prediction_label = ttk.Label(master, text="")
        self.prediction_label.place(x=500, y=350)

        self.graphical_plot_label = ttk.Label(master)
        self.graphical_plot_label.place(x=200, y=20)

        self.eeg_plot_label = ttk.Label(master)
        self.eeg_plot_label.place(x=200, y=400)

        self.l = ttk.Label(master, text="EEG Prediction")
        self.l.config(font=("Courier", 14))
        self.l.place(x=10, y=45)

        self.predict_button = ttk.Button(master, text="Run Prediction", command=self.run_prediction_loop)
        self.predict_button.place(x=10, y=100)

        self.index = 0
        self.model = None
        self.X_test_loaded = None
        self.predicted_data = None
        self.sock = None

        self.load_model()
        self.load_eeg_data()
        self.setup_bluetooth_connection()

    def load_model(self):
        try:
            self.model = pickle.load(open('EEGNet.pkl', 'rb'))
            # self.model = model
        except Exception as e:
            print(f"Error loading model: {e}")

    def load_eeg_data(self):
        try:
            with open('X.csv', 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                data = []
                original_shape = None

                for row in reader:
                    if 'Original Shape:' in row:
                        original_shape = tuple(map(int, row[1].strip('()').split(',')))
                    else:
                        data.append([float(val) for val in row])

                if original_shape:
                    self.X_test_loaded = np.array(data).reshape(original_shape)
                    self.X_test_loaded = self.X_test_loaded[:4]
                    self.predicted_data = self.model.predict(self.X_test_loaded).argmax(axis=-1)
        except Exception as e:
            print(f"Error loading EEG data: {e}")

    def setup_bluetooth_connection(self):
        try:
            esp32 = "ESP32test"
            address = "A0:A3:B3:AB:89:BA"
            devices = bluetooth.discover_devices()

            for addr in devices:
                if esp32 == bluetooth.lookup_name(addr):
                    address = addr
                    break

            port = 1
            self.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.sock.connect((address, port))
        except Exception as e:
            print(f"Bluetooth Error: {e}")

    def run_prediction_loop(self):
        self.index = 0  # Reset index to 0
        self.run_prediction()

    def run_prediction(self):
        if self.index < len(self.predicted_data):
            pred = self.predicted_data[self.index]
            sample = self.X_test_loaded[self.index].squeeze()

            try:
                if pred == 0:
                    print("Forward")
                    self.prediction_label.config(text=f'Prediction: Forward')
                elif pred == 1:
                    print("Left")
                    self.prediction_label.config(text=f'Prediction: Left')
                elif pred == 2:
                    print("Right")
                    self.prediction_label.config(text=f'Prediction: Right')
                elif pred == 3:
                    print("Reverse")
                    self.prediction_label.config(text=f'Prediction: Reverse')
                self.sock.send(str(pred + 1))
            except Exception as e:
                print(f"Exception Occurred: {e}")

            time.sleep(3)

            # Plot EEG data
            plt.figure(figsize=(6, 4))
            plt.plot(sample)
            plt.xlabel('Time Step')
            plt.ylabel('EEG Value')
            plt.title(f'Sample {self.index} EEG Data')
            plt.savefig("temp_plot.png")
            plt.close()

            # Update graphical plot
            img = Image.open("temp_plot.png")
            img = img.resize((400, 300), Image.BILINEAR)
            img = ImageTk.PhotoImage(img)
            self.graphical_plot_label.config(image=img)
            self.graphical_plot_label.image = img

            # Plot EEG data with color bar
            plt.figure(figsize=(10, 6))
            plt.imshow(sample, aspect='auto', cmap='viridis')
            plt.colorbar(label='EEG Value')
            plt.xlabel('Time Step')
            plt.ylabel('Sample')
            plt.title(f'EEG Data - Prediction: {pred}')
            plt.savefig("temp_plot_eeg.png")
            plt.close()

            img_eeg = Image.open("temp_plot_eeg.png")
            img_eeg = img_eeg.resize((400, 300), Image.BILINEAR)
            img_eeg = ImageTk.PhotoImage(img_eeg)
            self.eeg_plot_label.config(image=img_eeg)
            self.eeg_plot_label.image = img_eeg

            # Update index for next prediction
            self.index += 1

            # Call run_prediction after a delay of 3 seconds
            self.master.after(300, self.run_prediction)

root = tk.Tk()
app = EEGPredictionGUI(root)
root.mainloop()
