// ---- Motor driver (L298N) - belt only ----
int ENA = 9;
int IN1 = 8;
int IN2 = 7;

void setup() {
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // belt chalao (forward)
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 200);   // speed 0-255
}

void loop() {
  // kuch nahi - belt chalti rahegi
}
