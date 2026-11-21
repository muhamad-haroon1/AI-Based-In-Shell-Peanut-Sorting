// ================= CONTINUOUS BELT + SERVO SORTER =================
// Belt lagaatar chalti hai. Laptop (Python) serial pe bhejta hai:
//   'B' = BAD peanut  -> delay baad servo push kare
//   'G' = GOOD        -> kuch nahi
// NOTE: ENA pin 5 pe (Servo lib pin 9/10 ka PWM band karti)

#include <Servo.h>

int ENA = 5;
int IN1 = 8;
int IN2 = 7;

Servo myservo;
int SERVO_PIN = 10;

int BELT_SPEED  = 180;    // 0-255 (dheemi rakho taake blur kam ho)
int PUSH_DELAY  = 1500;   // ms: camera->servo peanut travel time (TUNE KARO)
int PUSH_ANGLE  = 90;     // push position
int REST_ANGLE  = 0;      // normal position
int PUSH_HOLD   = 500;    // ms servo pushed rahe

void setup() {
  Serial.begin(9600);
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // belt continuously chalao
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, BELT_SPEED);

  myservo.attach(SERVO_PIN);
  myservo.write(REST_ANGLE);
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'B') {                 // BAD peanut
      delay(PUSH_DELAY);            // peanut ko servo tak aane do
      myservo.write(PUSH_ANGLE);    // push (reject)
      delay(PUSH_HOLD);
      myservo.write(REST_ANGLE);    // wapas
    }
    // 'G' = good -> kuch nahi, pass hone do
  }
}
