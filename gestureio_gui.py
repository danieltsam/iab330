import asyncio
import threading
import struct
import csv
from datetime import datetime, timezone, timedelta
import tkinter as tk
from tkinter import ttk, messagebox

from bleak import BleakClient, BleakScanner
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === BLE UUIDs ===
HAR_SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
ACCEL_GYRO_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"
COMMAND_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef2"
STATUS_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef3"

AEST = timezone(timedelta(hours=10))  # Brisbane time

class GestureIOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GestureIO — HAR Data Stream")
        self.client = None
        self.devices = []
        self.streaming = False
        self.csv_writer = None
        self.csv_file = None

        self.data_buffer = [[], [], [], []]  # timestamp, ax, ay, az
        self.loop = asyncio.new_event_loop()

        # Build UI
        self._build_ui()

        # Start asyncio loop in background
        threading.Thread(target=self._run_loop, daemon=True).start()

    # --------------------------
    # GUI SETUP
    # --------------------------
    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="BLE Device:").grid(row=0, column=0, sticky="w")

        self.device_box = ttk.Combobox(frame, state="readonly", width=35)
        self.device_box.grid(row=0, column=1, padx=5)

        self.scan_button = ttk.Button(frame, text="Scan", command=self.scan_devices)
        self.scan_button.grid(row=0, column=2, padx=5)

        ttk.Button(frame, text="Connect", command=self.connect_device).grid(row=0, column=3, padx=5)

        self.status_label = ttk.Label(frame, text="Disconnected", foreground="red")
        self.status_label.grid(row=0, column=4, padx=10)

        self.stream_button = ttk.Button(frame, text="Start Stream", command=self.toggle_stream, state="disabled")
        self.stream_button.grid(row=1, column=0, columnspan=2, pady=5)

        self.record_button = ttk.Button(frame, text="Save Recording", command=self.toggle_recording, state="disabled")
        self.record_button.grid(row=1, column=2, columnspan=2, pady=5)

        # Matplotlib plot
        fig, self.ax = plt.subplots(figsize=(7, 3))
        self.lines = [self.ax.plot([], [])[0] for _ in range(3)]
        self.ax.set_title("Realtime Accelerometer Data (ax, ay, az)")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Value")
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=5, pady=10)

        self.root.after(200, self.update_plot)

    # --------------------------
    # ASYNC EVENT LOOP
    # --------------------------
    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # --------------------------
    # BLE SCANNING
    # --------------------------
    def scan_devices(self):
        """Disable input during scan and re-enable after."""
        self.device_box["state"] = "disabled"
        self.scan_button["state"] = "disabled"
        self.status_label.config(text="Scanning...", foreground="orange")
        self.root.update_idletasks()

        async def async_scan():
            try:
                self.devices = await BleakScanner.discover(timeout=5)
                device_list = [f"{d.name or 'Unknown'} ({d.address})" for d in self.devices]

                if device_list:
                    self.device_box["values"] = device_list
                    self.device_box.current(0)
                    self.status_label.config(
                        text=f"Scan complete ({len(device_list)} found)", foreground="green"
                    )
                else:
                    self.device_box["values"] = []
                    self.status_label.config(text="No devices found", foreground="red")
                    messagebox.showinfo("Scan Complete", "No BLE devices found.")
            except Exception as e:
                messagebox.showerror("Scan Error", str(e))
                self.status_label.config(text="Scan failed", foreground="red")
            finally:
                self.device_box["state"] = "readonly"
                self.scan_button["state"] = "normal"
                self.root.update_idletasks()

        asyncio.run_coroutine_threadsafe(async_scan(), self.loop)

    # --------------------------
    # BLE CONNECTION
    # --------------------------
    def connect_device(self):
        selected = self.device_box.get()
        if not selected:
            messagebox.showwarning("No Device Selected", "Please select a device from the list.")
            return
        address = selected.split("(")[-1].strip(")")

        self.status_label.config(text="Connecting...", foreground="orange")

        async def async_connect():
            try:
                self.client = BleakClient(address)
                await self.client.connect()
                await asyncio.sleep(1)
                if self.client.is_connected:
                    self.status_label.config(text="Connected", foreground="green")
                    self.stream_button["state"] = "normal"
                    self.record_button["state"] = "normal"
                    print(f"✅ Connected to {address}")
                else:
                    raise Exception("Failed to connect.")
            except Exception as e:
                messagebox.showerror("Connection Error", str(e))
                self.status_label.config(text="Disconnected", foreground="red")

        asyncio.run_coroutine_threadsafe(async_connect(), self.loop)

    # --------------------------
    # STREAM CONTROL
    # --------------------------
    def toggle_stream(self):
        if not self.client or not self.client.is_connected:
            messagebox.showwarning("Not Connected", "Please connect to a device first.")
            return
        start = not self.streaming
        self.stream_button.config(text="⏹ Stop Stream" if start else "▶ Start Stream")

        async def async_toggle_stream():
            try:
                if start:
                    await self.client.start_notify(ACCEL_GYRO_CHAR_UUID, self.handle_notification)
                    await asyncio.sleep(0.3)
                    await self.client.write_gatt_char(COMMAND_CHAR_UUID, b"START")
                    self.streaming = True
                    print("▶ Streaming started.")
                else:
                    await self.client.write_gatt_char(COMMAND_CHAR_UUID, b"STOP")
                    await asyncio.sleep(0.2)
                    await self.client.stop_notify(ACCEL_GYRO_CHAR_UUID)
                    self.streaming = False
                    print("⏹ Streaming stopped.")
            except Exception as e:
                messagebox.showerror("Stream Error", str(e))
                self.streaming = False
                self.stream_button.config(text="▶ Start Stream")

        asyncio.run_coroutine_threadsafe(async_toggle_stream(), self.loop)

    # --------------------------
    # RECORDING CONTROL
    # --------------------------
    def toggle_recording(self):
        if not self.csv_writer:
            filename = f"gestureio_{datetime.now(AEST).strftime('%Y%m%d_%H%M%S')}.csv"
            self.csv_file = open(filename, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["timestamp (AEST)", "ax", "ay", "az", "gx", "gy", "gz"])
            self.record_button.config(text="🛑 Stop Recording")
            print(f"💾 Recording started: {filename}")
        else:
            self.csv_file.close()
            self.csv_writer = None
            self.csv_file = None
            self.record_button.config(text="💾 Save Recording")
            print("🛑 Recording stopped.")

    # --------------------------
    # HANDLE NOTIFICATION
    # --------------------------
    def handle_notification(self, sender, data):
        imu = struct.unpack("<6h", data)
        ax, ay, az, gx, gy, gz = imu
        ts = datetime.now(AEST).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        self.data_buffer[0].append(ts)
        self.data_buffer[1].append(ax)
        self.data_buffer[2].append(ay)
        self.data_buffer[3].append(az)

        if len(self.data_buffer[0]) > 100:
            for buf in self.data_buffer:
                buf.pop(0)

        if self.csv_writer:
            self.csv_writer.writerow([ts, ax, ay, az, gx, gy, gz])

    # --------------------------
    # REALTIME PLOT UPDATE
    # --------------------------
    def update_plot(self):
        for i, line in enumerate(self.lines):
            line.set_ydata(self.data_buffer[i + 1])
            line.set_xdata(range(len(self.data_buffer[i + 1])))
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        self.root.after(200, self.update_plot)

# --------------------------
# MAIN APP ENTRY
# --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureIOApp(root)
    root.mainloop()
