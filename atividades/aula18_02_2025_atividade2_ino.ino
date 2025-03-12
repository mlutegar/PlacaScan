#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "Iphone de André";  // Substitua pelo nome da sua rede Wi-Fi
const char* password = "12345678";  // Substitua pela senha do Wi-Fi

// Configuração MQTT
const char* mqtt_server = "test.mosquitto.org";  // Servidor MQTT público
const int mqtt_port = 1883;  // Porta padrão MQTT
const char* mqtt_topic = "iotbr/esp32";  // Tópico MQTT onde vamos publicar

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  delay(10);
  Serial.println("Conectando ao WiFi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }

  Serial.println("\nWiFi conectado!");
  Serial.print("Endereço IP: ");
  Serial.println(WiFi.localIP());
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Conectando ao MQTT...");
    if (client.connect("ESP32Client1")) {
      Serial.println("Conectado!");
    } else {
      Serial.print("Falha, rc=");
      Serial.print(client.state());
      Serial.println(" Tentando novamente em 5 segundos...");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
}

void loop() {
  // Verificar se o cliente MQTT está conectado
  if (!client.connected()) {
    Serial.println("MQTT desconectado! Tentando reconectar...");
    reconnect();
  }

  // Se já estiver conectado, publique a mensagem
if (client.connected()) {
    Serial.println("MQTT já conectado! Publicando mensagem...");
    
    int valorSensor = random(20, 40);
    String mensagem = "Grupo placas policias, numero aleatorio: " + String(valorSensor);
    
    Serial.println("Publicando: " + mensagem);
    
    if (client.publish(mqtt_topic, mensagem.c_str())) {
        Serial.println("✅ Publicação bem-sucedida!");
    } else {
        Serial.println("❌ Falha ao publicar no MQTT!");
    }
}
  

  // Mantém a conexão MQTT ativa
  client.loop();

  delay(10000);  // Aguarda 10 segundos antes de enviar a próxima leitura
}
