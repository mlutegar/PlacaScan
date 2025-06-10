https://www.canva.com/design/DAGp968ZEDE/Ysf2rYL_f00USe7ULTsD3w/edit?utm_content=DAGp968ZEDE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

# LPR Analysis Tools: Otimização de Pré-processamento para OCR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-2023-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0-red.svg)](https://opencv.org/)

## Visão Geral

Este repositório contém ferramentas de análise e validação para sistemas de reconhecimento de placas veiculares (LPR), com foco específico na **otimização de técnicas de pré-processamento de imagem** para maximizar a precisão de OCR em hardware embarcado com recursos limitados.

O projeto investiga sistematicamente o impacto de diferentes algoritmos de pré-processamento na precisão do reconhecimento óptico de caracteres (OCR) para placas brasileiras, estabelecendo um trade-off crítico entre precisão e eficiência computacional.

## Problema de Pesquisa

**Como alcançar a máxima precisão de OCR para placas brasileiras usando algoritmos de pré-processamento leves e rápidos o suficiente para hardware embarcado com recursos restritos?**

## Arquitetura do Sistema

```
Imagem Capturada → YOLO Detection → Pré-processamento → OCR (Tesseract) → Pós-processamento → Resultado Final
                      ↓                    ↓                 ↓                    ↓
                 Bounding Box      Filtros Otimizados    Extração de Texto    Validação
```

## Técnicas de Pré-processamento Investigadas

Este projeto avalia **8 técnicas** selecionadas por sua simplicidade computacional e eficácia potencial:

| Técnica | Descrição | Custo Computacional | Aplicação |
|---------|-----------|-------------------|-----------|
| **Original** | Imagem sem processamento | Mínimo | Baseline |
| **Grayscale** | Conversão para escala de cinza | Muito baixo | Redução de dimensionalidade |
| **Otsu** | Limiarização automática de Otsu | Baixo | Binarização adaptativa |
| **Adaptive** | Limiarização adaptativa | Baixo | Condições de iluminação variável |
| **Bilateral** | Filtragem bilateral | Médio | Redução de ruído preservando bordas |
| **Sharpened** | Aumento de nitidez | Baixo | Melhoria de definição |
| **Resized2x** | Redimensionamento 2x | Alto | Aumento de resolução |
| **Inverted** | Inversão de cores | Muito baixo | Melhoria de contraste |

## Ferramentas Incluídas

### 1. Video Analysis Tool (`video_analysis.py`)
- Processa vídeos aplicando todas as técnicas de pré-processamento
- Executa OCR em múltiplos limiares de confiança (0.0, 0.2, 0.4, 0.6, 0.8)
- Gera análise batch completa com 921 tentativas de OCR

### 2. Validation Tool (`analysis-tool.py`)
Interface gráfica para validação manual dos resultados:
- Visualização lado a lado: imagem original vs. melhor processada
- Input para ground truth (texto correto da placa)
- Avaliação de qualidade (Excellent/Good/Poor/Unreadable)
- Navegação por teclado (←/→/Enter/Space)
- Filtragem: All/Deduplicated/High Quality Only
- Exportação para análise estatística

**Funcionalidades principais:**
```python
# Filtros de deduplicação
def deduplicate_plates(self):
    time_window = 5.0  # segundos
    similarity_threshold = 0.8

# Filtro de alta qualidade
def filter_high_quality_plates(self):
    confidence_threshold = 0.7
    min_bbox_area = 2000  # pixels
```

### 3. Dataset Curator (`dataset-curator.py`)
Curador automático para criação de datasets validados:
- Filtragem por critérios de qualidade
- Limitação de placas por vídeo
- Cópia automática de arquivos (original + 8 versões processadas)
- Geração de metadados e estatísticas
- Exportação para CSV para análise acadêmica

## Resultados Principais

### Performance Geral
- **Dataset:** 65 placas brasileiras, 921 tentativas de OCR
- **Precisão placa completa:** 3.1%
- **Precisão por caractere:** 26.8%

### Melhores Métodos

| Método | Precisão Média | Melhor Configuração | Trade-off |
|--------|---------------|-------------------|-----------|
| **Resized2x** | 21.6% | 66.7% (limiar 0.8) | Alta precisão, alto custo (4x memória) |
| **Inverted** | 11.8% | 11.8% (limiar 0.4) | Precisão competitiva, custo quase nulo |

### Estratégias Identificadas

1. **Estratégia de Máxima Precisão:**
   - Método: Resized2x (limiar 0.8)
   - Precisão: 66.7%
   - Limitação: Alto custo computacional

2. **Estratégia Otimizada para Embarcados:**
   - Método: Inverted (limiar 0.4)
   - Precisão: 11.8%
   - Vantagem: Sobrecarga computacional praticamente nula

## Instalação

### Pré-requisitos
- Python 3.10+
- Tesseract OCR
- YOLO v8 nano model

### Setup
```bash
git clone [repository-url]
cd lpr-analysis-tools
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.\.venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Configuração do Tesseract (Windows)
```bash
# Baixar de: https://github.com/UB-Mannheim/tesseract/wiki
# Instalar em: C:\Program Files\Tesseract-OCR\
# Configurar no código:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Uso das Ferramentas

### 1. Análise de Vídeo
```bash
python video_analysis.py
# Processa vídeos da pasta 'videos/'
# Gera resultados em 'video_analysis_results/'
```

### 2. Validação Manual
```bash
python analysis-tool.py
# Interface gráfica para validação
# Carrega dados de 'video_analysis_results/'
# Salva validações em 'validation_results.json'
```

### 3. Curadoria de Dataset
```bash
python dataset-curator.py
# Cria dataset curado em 'curated_dataset/'
# Filtra por qualidade e remove duplicatas
# Gera CSVs para análise estatística
```

## Estrutura de Dados

### Formato de Resultados OCR
```json
{
  "filename": "plate_001",
  "video_name": "video1.mp4",
  "confidence": 0.87,
  "ocr_results": {
    "grayscale": {
      "0.0": {"text": "ABC1234", "confidence": 0.92},
      "0.4": {"text": "ABC1234", "confidence": 0.89}
    },
    "resized2x": {
      "0.8": {"text": "ABC1234", "confidence": 0.95}
    }
  }
}
```

### Formato de Validação
```json
{
  "filename": "plate_001",
  "ground_truth": "ABC1234",
  "quality": "Good",
  "ocr_accuracies": {
    "resized2x": {
      "0.8": {
        "ocr_text": "ABC1234",
        "confidence": 0.95,
        "is_correct": true
      }
    }
  }
}
```

## Análise Estatística

### Métricas Calculadas
- Precisão por método e limiar
- Distribuição de qualidade
- Estatísticas de confiança YOLO
- Análise de erro por caractere vs. placa completa

### Exports Disponíveis
- `detailed_results.csv`: Resultados completos por método/limiar
- `accuracy_by_method.csv`: Métricas de precisão sumarizadas
- `validation_summary.csv`: Resumo das validações manuais

## Principais Descobertas

### 1. Trade-off Crítico
A escolha do pré-processamento é um **compromisso fundamental** entre precisão e eficiência computacional.

### 2. Análise de Erros
- Precisão por caractere (26.8%) é **8.5x maior** que precisão da placa completa (3.1%)
- Erros em um único caractere causam falha total do reconhecimento
- Necessidade de algoritmos de correção probabilística

### 3. Requisito de Validação Adicional
- Alta taxa de falsos positivos (96.9%)
- Necessidade de mecanismos de validação (formato, banco de dados)

## Próximos Passos

1. **Análise Preditiva:** Implementar correspondência probabilística usando confiança por caractere
2. **Expansão do Sistema:** Integração com bancos de dados nacionais
3. **Testes em Campo:** Validação com forças policiais em condições reais
4. **Dataset Aprimorado:** Cobertura de condições operacionais diversificadas

## Contribuição Acadêmica

Este projeto fornece:
- Metodologia sistemática para avaliação de pré-processamento em LPR
- Dataset validado de placas brasileiras
- Análise quantitativa de trade-offs computacionais
- Ferramentas reproduzíveis para pesquisa em OCR embarcado

## Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

## Citação

```bibtex
@misc{lpr_preprocessing_2025,
  title={LPR em Tempo Real para Segurança Pública: Otimizando o Reconhecimento em Sistemas Embarcados},
  author={André Costa and Marceu Filho and Michel Lutegar},
  year={2025},
  note={Disciplina: Sistemas Embarcados e IOT}
}
```
