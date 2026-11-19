# CitizenTech · Visão Computacional para Manutenção Urbana

Módulo de visão computacional do protótipo **CitizenTech**: triagem automática de **fissuras em concreto** a partir de imagens e vídeos, para indicar quais locais merecem inspeção.

> **Aviso de escopo.** O sistema apoia a triagem de locais para inspeção. A presença de fissura na imagem, isoladamente, não permite determinar a gravidade do dano nem a segurança estrutural da edificação.

## Integrantes

| Nome | RM |
|------|----|
| Luís Fernando de Oliveira Salgado | RM561401 |
| Ricardo de Paiva Melo | RM565522 |
| Jonas Alaf | RM566479 |

Turma: 2TIAPF · FIAP · Global Solution 2026.2

## Problema

Prefeituras e equipes de manutenção precisam priorizar vistorias em estruturas urbanas. Inspecionar tudo manualmente é lento e caro. O módulo classifica imagens como **com fissura** ou **sem fissura** e marca trechos suspeitos em vídeo, deixando a decisão final para o inspetor.

## Dados

- Base: *Concrete Crack Images for Classification* (Özgenel e Sorguç, Mendeley Data, DOI 10.17632/5y9wdsg2zt.1).
- 40 mil imagens 227×227 (20 mil por classe), geradas a partir de 458 fotos de alta resolução de prédios do campus da METU.
- Amostra usada: 3 000 imagens por classe (6 000 no total), balanceada.
- **Divisão sem vazamento:** os nomes dos arquivos não identificam a foto original, então os recortes foram agrupados por aparência (K-Means sobre características de uma CNN) e a divisão treino/validação/teste (70/15/15) é feita **por grupo**. É uma aproximação, não uma garantia.

## Modelo

- **MobileNetV3-Small** pré-treinada no ImageNet, com saída de 1 neurônio.
- Transferência de aprendizado em duas etapas: cabeça com backbone congelado, depois ajuste fino de toda a rede com taxa de aprendizado menor.
- Pré-processamento: 224×224, normalização ImageNet; aumento de dados (giros, brilho, contraste, saturação) apenas no treino.
- **Limiar de confiança** escolhido na validação por F-beta (beta = 2), privilegiando recall: perder uma fissura custa mais que pedir uma vistoria a mais.
- ResNet18 treinada com o mesmo pipeline como comparação; MobileNetV3-Large entra na comparação de custo.

## Resultados

Os números são gerados ao executar o notebook (células de avaliação e desempenho) e ficam em `outputs/relatorio_breve.md` e `outputs/resultados.json`. Preencha aqui após rodar:

| Métrica (teste) | Valor |
|-----------------|-------|
| Precisão | [preencher] |
| Recall | [preencher] |
| F1 | [preencher] |
| Limiar escolhido | [preencher] |
| Latência por imagem (CPU) | [preencher] ms |
| Imagens/s (CPU, lote 32) | [preencher] |
| FPS no vídeo de demonstração | [preencher] |

A análise de erros usa Grad-CAM e estatísticas de brilho, contraste e textura para investigar falsos positivos (sombras, juntas, rugosidade) e falsos negativos (fissuras finas, pouco contraste).

## Integração com o CitizenTech

- `citizentech_crack_module.py`: classe `CrackDetector` com
  - `predict_image`: uma imagem → classe, confiança e flag de inspeção;
  - `scan_frame`: varre um quadro em blocos de 227 px e devolve os trechos suspeitos (localização aproximada, não detecção por caixas);
  - `process_video`: vídeo de entrada → vídeo anotado + estatísticas de FPS e latência.
- `app_citizentech.py`: app **Streamlit** com envio de imagem, envio de vídeo e ajuste do limiar.

## Como executar

**Notebook (Google Colab, GPU T4):**
1. Abra `CitizenTech_Visao_Computacional_Fissuras.ipynb`.
2. Ambiente de execução → Alterar tipo → T4 GPU.
3. Execute as células em ordem. A base é baixada automaticamente; se falhar, envie as pastas `Positive` e `Negative` para `/content/data/` e rode a célula de novo.
4. A última célula gera `outputs.zip` com modelo, módulo, app, figuras, vídeo de demonstração e relatório.

**App localmente:**
```bash
pip install streamlit torch torchvision opencv-python-headless pillow
streamlit run app_citizentech.py   # com crack_model.pt na mesma pasta
```

## Estrutura de entrega

```
CitizenTech_Visao_Computacional_Fissuras.ipynb
outputs/
├── crack_model.pt                 # pesos + limiar
├── citizentech_crack_module.py
├── app_citizentech.py
├── demo_video_entrada.mp4
├── demo_video_saida.mp4
├── relatorio_breve.md
├── resultados.json
└── fig*.png
```

## Limitações

- Base de um único campus: generalização para outras cidades, câmeras e iluminações não foi medida.
- Divisão por grupos estimados é aproximada.
- Localização por blocos de 227 px, não detecção por caixas delimitadoras.
- Não estima largura, extensão ou gravidade da fissura.
- O vídeo de demonstração é simulado a partir de recortes do conjunto de teste: serve para mostrar funcionamento e velocidade, não para medir acurácia.

## Referências

- Özgenel, Ç. F.; Sorguç, A. G. *Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings*. ISARC 2018.
- Zhang, L. et al. *Road Crack Detection Using Deep Convolutional Neural Network*. IEEE ICIP 2016.
