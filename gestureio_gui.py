import asyncio
import threading
import struct
import csv
import os
from datetime import datetime, timezone, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
from db import get_db, get_recent_predictions

from bleak import BleakClient, BleakScanner
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from live_inference import LiveCNNInference

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
        self.recording_active = False
        self.recording_data = []
        self.recording_filename = None
        self.current_recording_label = None

        self.data_buffer = [[], [], [], []]  # timestamp, ax, ay, az
        self.loop = asyncio.new_event_loop()

        self.inference_active = False
        self.infer = None  # type: LiveCNNInference | None
        self.last_pred_text = tk.StringVar(value="—")

        self._build_ui()
        threading.Thread(target=self._run_loop, daemon=True).start()

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

        self.record_button = ttk.Button(frame, text="Start Recording", command=self.toggle_recording, state="disabled")
        self.record_button.grid(row=1, column=2, columnspan=1, pady=5)

        ttk.Label(frame, text="Model Export Dir:").grid(row=2, column=0, sticky="w")
        self.export_dir_var = tk.StringVar(value="models/")
        self.export_entry = ttk.Entry(frame, textvariable=self.export_dir_var, width=35)
        self.export_entry.grid(row=2, column=1, columnspan=2, padx=5, sticky="ew")

        self.infer_button = ttk.Button(frame, text="Start Inference", command=self.toggle_inference, state="disabled")
        self.infer_button.grid(row=2, column=3, padx=5)

        ttk.Label(frame, text="File Name:").grid(row=3, column=0, sticky="w")
        self.filename_var = tk.StringVar()
        self.filename_entry = ttk.Entry(frame, textvariable=self.filename_var, width=35)
        self.filename_entry.grid(row=3, column=1, columnspan=3, padx=5, sticky="ew")
        self.filename_var.set("gesture_recordings.csv")

        ttk.Label(frame, text="Prediction:").grid(row=1, column=4, sticky="e")
        self.pred_label = ttk.Label(frame, textvariable=self.last_pred_text, font=("Segoe UI", 12, "bold"))
        self.pred_label.grid(row=1, column=5, sticky="w")

        fig, self.ax = plt.subplots(figsize=(7, 4))
        self.lines = [self.ax.plot([], [])[0] for _ in range(3)]
        self.ax.set_title("Realtime Accelerometer Data (ax, ay, az)")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Value")
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=6, pady=10, sticky="nsew")

        for c in range(6):
            self.root.grid_columnconfigure(c, weight=1)
        self.root.grid_rowconfigure(4, weight=1)

        self.root.after(200, self.update_plot)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def scan_devices(self):
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
                    self.infer_button["state"] = "normal"
                    print(f"Connected to {address}")
                else:
                    raise Exception("Failed to connect.")
            except Exception as e:
                messagebox.showerror("Connection Error", str(e))
                self.status_label.config(text="Disconnected", foreground="red")

        asyncio.run_coroutine_threadsafe(async_connect(), self.loop)

    def toggle_stream(self):
        if not self.client or not self.client.is_connected:
            messagebox.showwarning("Not Connected", "Please connect to a device first.")
            return
        start = not self.streaming
        self.stream_button.config(text="Stop Stream" if start else "Start Stream")

        async def async_toggle_stream():
            try:
                if start:
                    await self.client.start_notify(ACCEL_GYRO_CHAR_UUID, self.handle_notification)
                    await asyncio.sleep(0.3)
                    await self.client.write_gatt_char(COMMAND_CHAR_UUID, b"START")
                    self.streaming = True
                    print("Streaming started.")
                else:
                    await self.client.write_gatt_char(COMMAND_CHAR_UUID, b"STOP")
                    await asyncio.sleep(0.2)
                    await self.client.stop_notify(ACCEL_GYRO_CHAR_UUID)
                    self.streaming = False
                    print("Streaming stopped.")
            except Exception as e:
                messagebox.showerror("Stream Error", str(e))
                self.streaming = False
                self.stream_button.config(text="▶ Start Stream")

        asyncio.run_coroutine_threadsafe(async_toggle_stream(), self.loop)

    def toggle_inference(self):
        if not self.client or not self.client.is_connected:
            messagebox.showwarning("Not Connected", "Please connect to a device first.")
            return

        if not self.inference_active:
            export_dir = self.export_dir_var.get().strip()
            if not export_dir:
                messagebox.showwarning("Export Dir Required", "Enter the model export directory.")
                return
            try:
                self.infer = LiveCNNInference(export_dir=export_dir)
                self.infer.reset()
                self.inference_active = True
                self.infer_button.config(text="Stop Inference")
                print(f"Inference started (cnn_1d) from {export_dir}")
            except Exception as e:
                messagebox.showerror("Inference Error", str(e))
                self.inference_active = False
                self.infer = None
        else:
            self.inference_active = False
            if self.infer:
                self.infer.reset()
            self.infer = None
            self.last_pred_text.set("—")
            self.infer_button.config(text="Start Inference")
            print("Inference stopped.")

    def toggle_recording(self):
        if not self.recording_active:
            filename = self.filename_var.get().strip()
            if not filename:
                messagebox.showwarning("File Name Required", "Please enter a file name before recording.")
                return
            if not filename.lower().endswith(".csv"):
                filename = f"{filename}.csv"

            self.recording_filename = filename
            self.current_recording_label = datetime.now(AEST).strftime("Recording %Y-%m-%d %H:%M:%S")
            self.recording_data = []
            self.recording_active = True
            self.record_button.config(text="Stop Recording")
            print(f"Recording started: {filename}")
        else:
            self._finalize_recording()

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

        if self.inference_active and self.infer is not None:
            try:
                out = self.infer.update([ax, ay, az, gx, gy, gz])
                if out is not None:
                    label, probs = out
                    conf = probs.get(label, 0.0)
                    self.root.after(0, lambda: self.last_pred_text.set(f"{label}  ({conf:.2f})"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Inference Error", str(e)))
                self.inference_active = False
                self.infer = None
                self.root.after(0, lambda: self.infer_button.config(text="Start Inference"))

        if self.recording_active:
            self.recording_data.append(f"{ts},{ax},{ay},{az},{gx},{gy},{gz}")
            if len(self.recording_data) >= 128:
                self.recording_active = False
                self.root.after(0, self._finalize_recording)

    def update_plot(self):
        for i, line in enumerate(self.lines):
            line.set_ydata(self.data_buffer[i + 1])
            line.set_xdata(range(len(self.data_buffer[i + 1])))
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        self.root.after(200, self.update_plot)

    def _save_recording(self):
        if not self.recording_filename:
            return

        if not self.recording_data:
            print("️ No data captured during recording; nothing to save.")
            return

        filename = self.recording_filename
        column_label = self.current_recording_label or datetime.now(AEST).strftime(
            "Recording %Y-%m-%d %H:%M:%S"
        )

        rows = []
        if os.path.exists(filename):
            with open(filename, newline="") as existing_file:
                rows = list(csv.reader(existing_file))

        if not rows:
            rows = [[column_label]]
        else:
            existing_columns = len(rows[0]) if rows else 0
            for row in rows:
                if len(row) < existing_columns:
                    row.extend([""] * (existing_columns - len(row)))
                elif len(row) > existing_columns:
                    del row[existing_columns:]
            rows[0].append(column_label)

        total_columns = len(rows[0])

        for row in rows[1:]:
            if len(row) < total_columns:
                row.extend([""] * (total_columns - len(row)))
            elif len(row) > total_columns:
                del row[total_columns:]

        for index, sample in enumerate(self.recording_data, start=1):
            if index >= len(rows):
                rows.append([""] * total_columns)
            if len(rows[index]) < total_columns:
                rows[index].extend([""] * (total_columns - len(rows[index])))
            rows[index][total_columns - 1] = sample

        for row in rows:
            if len(row) < total_columns:
                row.extend([""] * (total_columns - len(row)))

        with open(filename, "w", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerows(rows)

        print(f"Recording saved to {os.path.abspath(filename)} (column: {column_label})")

    def _finalize_recording(self):
        self.recording_active = False
        sample_count = len(self.recording_data)

        if self.record_button.cget("text") != "Start Recording":
            self.record_button.config(text="Start Recording")

        try:
            self._save_recording()
        except Exception as e:
            messagebox.showerror("Save Error", str(e))
        finally:
            self.recording_data = []
            self.recording_filename = None
            self.current_recording_label = None
            if sample_count:
                print(f"Recording stopped after capturing {sample_count} samples.")
class DBPanel:
    """
    Simple MongoDB viewer: Connect, Refresh, and (optionally) filter by session_id.
    """
    def __init__(self, root: tk.Tk):
        self.root = root
        self.frame = ttk.LabelFrame(root, text="MongoDB — Recent Predictions")
        self.frame.grid(row=5, column=0, columnspan=6, sticky="nsew", padx=8, pady=8)

        controls = ttk.Frame(self.frame)
        controls.pack(fill="x", padx=6, pady=6)

        self.status = tk.StringVar(value="Disconnected")
        ttk.Button(controls, text="Connect DB", command=self.connect_db).pack(side="left")
        ttk.Button(controls, text="Refresh", command=self.refresh).pack(side="left", padx=6)


        ttk.Label(controls, text="session_id:").pack(side="left", padx=(12, 4))
        self.session_entry = ttk.Entry(controls, width=40)
        self.session_entry.pack(side="left")
        self.use_current_btn = ttk.Button(controls, text="Use Current Session", command=self._use_current_session)
        self.use_current_btn.pack(side="left", padx=6)

        ttk.Label(controls, textvariable=self.status).pack(side="left", padx=12)

  
        cols = ("created_at", "gesture", "score", "session_id")
        self.tree = ttk.Treeview(self.frame, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (170, 140, 80, 280)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        
        yscroll = ttk.Scrollbar(self.frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        yscroll.place(in_=self.tree, relx=1.0, rely=0, relheight=1.0, x=-1)

        self._connected = False
        self._session_getter = None   # optional hook to read app.infer.session_id

        
        self.root.grid_rowconfigure(5, weight=1)

    def link_session_getter(self, getter):
        """Pass a function that returns the current session_id (e.g., lambda: app.infer.session_id)."""
        self._session_getter = getter

    def _use_current_session(self):
        try:
            sid = self._session_getter() if self._session_getter else ""
            if sid:
                self.session_entry.delete(0, tk.END)
                self.session_entry.insert(0, sid)
                self.refresh()
            else:
                messagebox.showinfo("MongoDB", "No active session_id found (start inference first).")
        except Exception as e:
            messagebox.showerror("MongoDB", f"Could not get current session_id:\n{e}")

    def connect_db(self):
        try:
       
            get_db().command("ping")
            self._connected = True
            self.status.set("Connected")
            self.refresh()
        except Exception as e:
            self._connected = False
            self.status.set("Disconnected")
            messagebox.showerror("MongoDB", f"Could not connect to MongoDB:\n{e}")

    def refresh(self):
        if not self._connected:
            messagebox.showinfo("MongoDB", "Not connected yet. Click 'Connect DB' first.")
            return
        try:
     
            sid = self.session_entry.get().strip()
            if sid:
                db = get_db()
                rows = list(db.predictions.find({"session_id": sid}).sort("created_at", -1).limit(10))
            else:
                rows = get_recent_predictions(limit=10)

       
            for i in self.tree.get_children():
                self.tree.delete(i)

            for doc in rows:
                ts = doc.get("created_at")
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
                gesture = doc.get("gesture", "")
                score = f"{doc.get('score', 0):.2f}"
                session = doc.get("session_id", "")
                self.tree.insert("", "end", values=(ts_str, gesture, score, session))

            self.status.set(f"Connected • {len(rows)} rows")
        except Exception as e:
            messagebox.showerror("MongoDB", f"Failed to fetch predictions:\n{e}")

    def start_auto_refresh(self, ms=2000):
        def _tick():
            if self._connected:
                try:
                    self.refresh()
                except Exception:
                    pass
            self.root.after(ms, _tick)
        self.root.after(ms, _tick)


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureIOApp(root)
    db_panel = DBPanel(root)
    root.mainloop()