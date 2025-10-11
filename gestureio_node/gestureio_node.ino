#include <ArduinoBLE.h>
#include <Arduino_LSM6DS3.h> // IMU sensor (Nano 33 IoT built-in)

// === BLE UUIDs ===
#define SERVICE_UUID          "12345678-1234-5678-1234-56789abcdef0"
#define ACCELGYRO_CHAR_UUID   "12345678-1234-5678-1234-56789abcdef1"
#define COMMAND_CHAR_UUID     "12345678-1234-5678-1234-56789abcdef2"
#define STATUS_CHAR_UUID      "12345678-1234-5678-1234-56789abcdef3"

// === BLE Objects ===
BLEService harService(SERVICE_UUID);
BLECharacteristic accelGyroChar(ACCELGYRO_CHAR_UUID, BLERead | BLENotify, 12);
BLEStringCharacteristic commandChar(COMMAND_CHAR_UUID, BLEWrite, 10);
BLEStringCharacteristic statusChar(STATUS_CHAR_UUID, BLERead | BLENotify, 10);

// === System State ===
bool isStreaming = false;
unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_MS = 20; // 50 Hz sampling rate

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("🚀 GestureIO Node Initializing...");

  if (!IMU.begin()) {
    Serial.println("❌ Failed to initialize IMU!");
    while (1);
  }

  if (!BLE.begin()) {
    Serial.println("❌ Starting BLE failed!");
    while (1);
  }

  // BLE Configuration
  BLE.setLocalName("GestureIO_Node");
  BLE.setAdvertisedService(harService);

  harService.addCharacteristic(accelGyroChar);
  harService.addCharacteristic(commandChar);
  harService.addCharacteristic(statusChar);
  BLE.addService(harService);

  statusChar.writeValue("IDLE");

  commandChar.setEventHandler(BLEWritten, onCommandReceived);

  BLE.advertise();
  Serial.println("✅ BLE GestureIO Node ready and advertising...");
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("🔗 Connected to: ");
    Serial.println(central.address());

    unsigned long connectTime = millis();

    while (central.connected()) {
      BLE.poll();

      // Allow some time for the central to subscribe to notifications
      if ((millis() - connectTime) < 5000 && !isStreaming) {
        delay(10);
        continue;
      }

      if (isStreaming && millis() - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        lastSampleTime = millis();
        sendIMUSample();
      }
    }

    // Reset when disconnected
    isStreaming = false;
    statusChar.writeValue("IDLE");
    Serial.println("❌ Disconnected.");
  }
}

// === Command Handler ===
void onCommandReceived(BLEDevice central, BLECharacteristic characteristic) {
  String cmd = commandChar.value();
  cmd.trim();

  Serial.print("📥 Command received: ");
  Serial.println(cmd);

  if (cmd.equalsIgnoreCase("START")) {
    isStreaming = true;
    statusChar.writeValue("STREAMING");
    Serial.println("▶️ Streaming started.");
  } 
  else if (cmd.equalsIgnoreCase("STOP")) {
    isStreaming = false;
    statusChar.writeValue("IDLE");
    Serial.println("⏹ Streaming stopped.");
  } 
  else {
    Serial.print("⚠️ Unknown command: ");
    Serial.println(cmd);
  }
}

// === IMU Sampling ===
void sendIMUSample() {
  float ax, ay, az, gx, gy, gz;

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    // Convert to int16_t with scaling (mg, mdps)
    int16_t ax_i = (int16_t)(ax * 1000);
    int16_t ay_i = (int16_t)(ay * 1000);
    int16_t az_i = (int16_t)(az * 1000);
    int16_t gx_i = (int16_t)(gx * 1000);
    int16_t gy_i = (int16_t)(gy * 1000);
    int16_t gz_i = (int16_t)(gz * 1000);

    byte packet[12];
    memcpy(packet, &ax_i, 2);
    memcpy(packet + 2, &ay_i, 2);
    memcpy(packet + 4, &az_i, 2);
    memcpy(packet + 6, &gx_i, 2);
    memcpy(packet + 8, &gy_i, 2);
    memcpy(packet + 10, &gz_i, 2);

    accelGyroChar.writeValue(packet, 12);
  }
}
