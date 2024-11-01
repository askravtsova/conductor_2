#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

// Create an instance of the BNO055 sensor
Adafruit_BNO055 bno = Adafruit_BNO055(55);

void setup() {
  // Start the serial communication
  Serial.begin(9600);

  // Initialize the BNO055 sensor
  if (!bno.begin()) {
    Serial.println("No BNO055 detected. Check wiring or I2C address!");
    while (1);
  }

  // Calibrate the BNO055 (optional external crystal)
  bno.setExtCrystalUse(true);

  // Wait for the sensor to be fully calibrated
  while (!isFullyCalibrated()) {
    printCalibrationStatus();
    delay(500);  // Check calibration every half second
  }

  Serial.println("Calibration complete! Ready to start data collection...");
  delay(1000); // Small delay before starting data collection
}

void loop() {
  unsigned long duration = 10000;  // 10 seconds duration for each sample

  // Collect 5 samples, each lasting 10 seconds
  for (int sampleNum = 1; sampleNum <= 5; sampleNum++) {
    Serial.print("Starting data collection for sample ");
    Serial.println(sampleNum);

    // Print CSV headers for easier data processing
    Serial.println("Time(ms),Orientation_X,Orientation_Y,Orientation_Z,Angular_Velocity_X,Angular_Velocity_Y,Angular_Velocity_Z,Linear_Acceleration_X,Linear_Acceleration_Y,Linear_Acceleration_Z");

    unsigned long startMillis = millis();
    unsigned long currentMillis;

    // Record data for 10 seconds
    while ((currentMillis = millis()) - startMillis < duration) {
      // Get orientation (Euler angles)
      imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);

      // Get angular velocity
      imu::Vector<3> angVel = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);

      // Get linear acceleration (without gravity)
      imu::Vector<3> linAccel = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);

      // Timestamp in milliseconds
      unsigned long timeStamp = millis();

      // Print data to serial in CSV format
      Serial.print(timeStamp); Serial.print(",");
      Serial.print(euler.x()); Serial.print(",");
      Serial.print(euler.y()); Serial.print(",");
      Serial.print(euler.z()); Serial.print(",");
      Serial.print(angVel.x()); Serial.print(",");
      Serial.print(angVel.y()); Serial.print(",");
      Serial.print(angVel.z()); Serial.print(",");
      Serial.print(linAccel.x()); Serial.print(",");
      Serial.print(linAccel.y()); Serial.print(",");
      Serial.println(linAccel.z());

      delay(10);  // 100Hz data rate (collect data every 10ms)
    }

    // Signal the end of data collection for this sample
    Serial.print("Data collection for sample ");
    Serial.print(sampleNum);
    Serial.println(" complete.");

    // Pause for 1 second before starting the next sample (if there are more)
    if (sampleNum < 5) {
      delay(1000);
    }
  }

  // Stop program after completing all samples
  Serial.println("All samples collected. Stopping.");
  while (1);  // Stop the program
}

// Function to check if the sensor is fully calibrated
bool isFullyCalibrated() {
  uint8_t system, gyro, accel, mag;
  bno.getCalibration(&system, &gyro, &accel, &mag);
  return (system == 3 && gyro == 3 && accel == 3 && mag == 3);  // Fully calibrated when all are at 3
}

// Function to print calibration status
void printCalibrationStatus() {
  uint8_t system, gyro, accel, mag;
  bno.getCalibration(&system, &gyro, &accel, &mag);
  Serial.print("System Calib: "); Serial.print(system, DEC);
  Serial.print(" Gyro Calib: "); Serial.print(gyro, DEC);
  Serial.print(" Accel Calib: "); Serial.print(accel, DEC);
  Serial.print(" Mag Calib: "); Serial.println(mag, DEC);
}
