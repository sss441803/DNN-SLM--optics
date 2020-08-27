int inByte;
int CLK = 8;
int CW = 9;
int ENn = 4;
int ENp = 11;
int Fan = 6;
int Peltier = 3;

void setup(){
  Serial.begin(9600);
  Serial.println("Ready");
  pinMode(CLK, OUTPUT);
  pinMode(CW, OUTPUT);
  pinMode(ENp, OUTPUT);
  pinMode(ENn, OUTPUT);
  digitalWrite(CLK, LOW);
  digitalWrite(CW, LOW);
  digitalWrite(ENp, LOW);
  digitalWrite(ENn, HIGH);
  analogWrite(Fan, 0);
  analogWrite(Peltier, 0);
}

void peltier(int dir){
  if (dir == '4'){
    analogWrite(Peltier, 100);
    }
  else {
    analogWrite(Peltier, 0);
    }
  }

void fan(int dir){
  if (dir == '2'){
    analogWrite(Fan, 100);
    Serial.write("fan on");
    }
  else {
    analogWrite(Fan, 0);
    }
  }

void motion(int duration, int dir) {
  if (dir=='1') {
    digitalWrite(CW, HIGH);
    }
  else {
    digitalWrite(CW, LOW);
    }
  digitalWrite(ENp, HIGH);
  digitalWrite(ENn, LOW);
  delay(300);
  for (int i = 0; i <= duration; i++){
    digitalWrite(CLK, HIGH);
    delayMicroseconds(30);
    digitalWrite(CLK, LOW);
    delayMicroseconds(100);
  }
}

void stop() {
  digitalWrite(ENn, HIGH);
  digitalWrite(ENp, LOW);
  digitalWrite(CLK, LOW);
  digitalWrite(CW, LOW);
}

void loop() {
  if(Serial.available()){
    inByte = Serial.read();
    if (inByte == '1' or inByte == '0'){
    motion(200, inByte);
    stop();
    }
    else if (inByte == '2' or inByte == '3'){
    fan(inByte);
    }
    else if (inByte == '4' or inByte == '5'){
    peltier(inByte);
    }
    Serial.write(inByte);
  }
  while(Serial.available()){Serial.read();}
  delay(1000);
}
