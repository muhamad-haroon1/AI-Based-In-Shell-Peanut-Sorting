#include <Servo.h>

// ---- Motor driver (L298N) ----
int ENA = 5;    // NOTE: pin 9 se 5 pe move kiya (Servo lib pin 9/10 ka PWM band karti)
int IN1 = 8;
int IN2 = 7;

// ---- Servo ----
Servo myservo;
int SERVO_PIN = 10;

void setup() {
  // motor pins
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // belt chalao (forward)
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 200);    // speed 0-255

  // servo
  myservo.attach(SERVO_PIN);
  myservo.write(0);         // start at 0 degree
  delay(500);
}

void loop() {
  delay(1000);              // 1 second wait
  myservo.write(90);        // 90 degree
  delay(1000);
  myservo.write(0);         // back to 0
  delay(1000);
}
