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

## Instalação

```bash
# Instruções de instalação serão adicionadas em breve
```

## Uso

### Detector básico

```python
# Exemplos de uso do detector básico serão adicionados em breve
```

### Pipeline completa

```python
# Exemplos de uso da pipeline completa serão adicionados em breve
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

_A estrutura detalhada do projeto será adicionada em breve_

## Trabalhos Futuros

- Expandir o dataset com mais placas brasileiras
- Implementar detecção de adulterações em placas
- Otimizar para dispositivos móveis

## Contribuições

_Instruções para contribuições serão adicionadas em breve_

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Agradecimentos

- Professor Rigel pelo suporte e orientação
- Professores e colegas do curso
- Comunidade YOLOv8 pelos excelentes recursos
<!-- - Contribuidores dos projetos OpenCV, Tesseract e EasyOCR -->
