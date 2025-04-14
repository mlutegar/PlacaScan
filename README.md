# IBMEC-SistemasEmbarcados

Esse repositório é destinado ao desenvolvimento de um produto baseado nos conhecimentos adquiridos na matéria Sistemas Embarcados.

# PlacaScan: Sistema de Reconhecimento de Placas Veiculares

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-2023-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0-red.svg)](https://opencv.org/)

## Projeto Aplicado

Este projeto consiste no desenvolvimento de um sistema embarcado para uso pelas forças policiais do Rio de Janeiro. O sistema integra um leitor automático de placas de veículos em tempo real, permitindo a verificação instantânea da situação veicular. Ele identifica possíveis irregularidades, incluindo restrições administrativas, mandados de busca e apreensão, e registros de furto ou roubo. Com essa tecnologia, a fiscalização torna-se mais eficiente, contribuindo para a segurança pública e a agilidade no combate a crimes relacionados a veículos.

## Implementação

Estamos usando o modelo YOLO V8 nano como algoritmo para reconhecimento das placas.

## Visão Geral

PlacaScan é um sistema inteligente para detecção e reconhecimento de placas veiculares, desenvolvido como projeto acadêmico. O sistema utiliza visão computacional avançada e técnicas de inteligência artificial para identificar placas em imagens e vídeos e extrair o texto usando OCR.

![Demo do Sistema](imagens/projeto1.jpeg)

## Características

- **Detecção em tempo real** de placas veiculares em streams de vídeo
- **Reconhecimento óptico de caracteres (OCR)** para extração do texto da placa
- **Interface intuitiva** para visualização dos resultados
- Suporte para diferentes formatos de placas (com foco em padrões brasileiros)

## Tecnologias

- **YOLOv8 nano**: Framework para detecção de objetos em tempo real
- **Python 3.10**: Linguagem de programação principal
<!-- - **OpenCV 4.7.0**: Processamento de imagens e manipulação de vídeo
- **Tesseract & EasyOCR**: Motores de reconhecimento de caracteres
- **SQLite**: Armazenamento local de dados e resultados// -->

## Instalação do PlacaScan

Siga estas etapas para configurar o ambiente de desenvolvimento do PlacaScan:

### 1. Clone o Repositório
Abra seu terminal ou prompt de comando e clone este repositório para sua máquina local. Certifique-se de ter o [Git](https://git-scm.com/) instalado.

```bash
git clone [https://github.com/SEU_USUARIO/IBMEC-SistemasEmbarcados.git](https://github.com/SEU_USUARIO/IBMEC-SistemasEmbarcados.git) # Substitua pelo URL correto do seu repositório
cd IBMEC-SistemasEmbarcados # Ou o nome da pasta onde o código do PlacaScan está localizado
```

### 2. Crie um Ambiente Virtual
É altamente recomendado usar um ambiente virtual para isolar as dependências do projeto. Navegue até a pasta do projeto clonado e crie um ambiente virtual (vamos chamá-lo de `.venv`).

```bash
python -m venv .venv
```
*Observação: Certifique-se de estar usando Python 3.10 ou compatível, conforme especificado no README.*

### 3. Ative o Ambiente Virtual
* **No Windows:**
    ```bash
    .\.venv\Scripts\activate
    ```
* **No macOS/Linux:**
    ```bash
    source .venv/bin/activate
    ```
Após a ativação, você verá o nome do ambiente (`.venv`) no início do prompt do seu terminal.

### 4. Instale as Dependências
Este projeto utiliza um arquivo `requirements.txt` para listar todas as bibliotecas Python necessárias. Certifique-se de que este arquivo (`requirements.txt`) esteja presente na raiz do projeto e contenha pelo menos as seguintes dependências (e quaisquer outras que você adicionar):

* `ultralytics` (para YOLOv8)
* `opencv-python` (para OpenCV)
* `pytesseract` (para OCR)
* *Outras dependências como `easyocr` se você quiser usar outros métodos de OCR*

Execute o seguinte comando para instalar todas as dependências listadas no arquivo:

```bash
pip install -r requirements.txt
```

### 5. Instale o Tesseract OCR no Windows
Como este projeto utiliza o pytesseract, é necessário instalar o Tesseract OCR no seu sistema:

* Baixe o instalador do Tesseract para Windows do site oficial: https://github.com/UB-Mannheim/tesseract/wiki
* Execute o instalador e siga as instruções (recomenda-se manter o caminho de instalação padrão: `C:\Program Files\Tesseract-OCR`)
* Adicione o caminho do Tesseract à variável de ambiente PATH do sistema ou especifique o caminho diretamente no código
* Verifique a instalação executando `tesseract --version` no prompt de comando

## Uso do PlacaScan

### Pipeline completa

Para utilizar a pipeline completa com reconhecimento de texto:

1. Certifique-se de que o Tesseract OCR está instalado corretamente em seu sistema
2. Configure sua webcam e conecte ao computador
3. Execute o arquivo **pipeline-completa.py**:
   ```bash
   python pipeline-completa.py
   ```
4. Aponte a câmera para placas de veículos
5. O sistema irá:
   - Detectar as placas em tempo real
   - Reconhecer o texto das placas usando OCR
   - Salvar imagens das placas na pasta **placas_crops**
   - Gerar arquivos TXT com o texto reconhecido para cada placa
6. O texto reconhecido será exibido no terminal em tempo real
7. Pressione ESC para encerrar o programa

```python
import os
import cv2
import pytesseract
from ultralytics import YOLO

# Configuração do caminho para o executável do Tesseract (necessário no Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Pasta de saída
save_dir = 'placas_crops'
os.makedirs(save_dir, exist_ok=True)

# Carrega modelo
model = YOLO('placa-veicular-model.pt')

cap = cv2.VideoCapture(0)
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        crop = frame[y1:y2, x1:x2]

        # --- OCR ---
        # Configurações: PSM 7 (uma linha), whitelist só A–Z e 0–9
        config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(crop, config=config).strip()
        print(f"[{counter:04d}] Placa reconhecida:", text)

        # Salva imagem e TXT
        img_path  = os.path.join(save_dir, f'placa_{counter:04d}.jpg')
        txt_path  = os.path.join(save_dir, f'placa_{counter:04d}.txt')
        cv2.imwrite(img_path, crop)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        counter += 1
        # desenha retângulo pra debug
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow('Detecção + OCR', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
```

## Desempenho

_Métricas de desempenho serão adicionadas em breve_

## Cronograma de Desenvolvimento

| Data | Atividade | Descrição |
|------|-----------|-----------|
| 18/02/2025 | Aula inaugural | Introdução à disciplina e discussão de possibilidades de projetos |
| 25/02/2025 | Definição do projeto | Seleção do tema e estabelecimento do objetivo de submissão ao SBRT |
| 04/03/2025 | Implementação inicial | Desenvolvimento da estrutura base utilizando YOLOv8 |
| 07/03/2025 | Desenvolvimento BD e treinamento | Configuração do armazenamento e treinamento do modelo (3 dias) |
| 11/03/2025 | Testes em tempo real | Avaliação de desempenho do sistema com webcam |
| 18/03/2025 | Integração OCR | Implementação do pipeline completo de reconhecimento |

## Estrutura do Projeto

A estrutura do projeto PlacaScan é organizada da seguinte forma:

projeto-sistema-embarcados-policial/
├── .idea/                     # Pasta de configuração do IDE
├── runs/detect/               # Pasta contendo resultados das detecções
├── .gitignore                 # Arquivo de configuração Git (arquivos ignorados)
├── COMMANDS.md                # Documentação dos comandos do projeto
├── placa-veicular-model.pt    # Modelo YOLOv8 treinado para detectar placas veiculares
├── requirements.txt           # Lista de dependências Python do projeto
└── yolo-placa.py              # Script principal de detecção de placas veiculares

A pasta runs/detect armazena os resultados das detecções realizadas pelo sistema. O arquivo placa-veicular-model.pt contém o modelo YOLOv8 treinado especificamente para detectar placas veiculares brasileiras. O script principal yolo-placa.py implementa a lógica de detecção e reconhecimento de placas usando a webcam.

## Trabalhos Futuros

- Expandir o dataset com mais placas brasileiras
- Implementar detecção de adulterações em placas
- Otimizar para dispositivos móveis

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Agradecimentos

- Professor Rigel pelo suporte e orientação
- Colegas do curso
- Comunidade YOLOv8 pelos excelentes recursos
- Contribuidores dos projetos Tesseract
