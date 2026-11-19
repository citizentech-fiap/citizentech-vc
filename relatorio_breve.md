# Relatório breve — Visão computacional para manutenção urbana (CitizenTech)

> O sistema apoia a triagem de locais para inspeção. A presença de fissura na imagem, isoladamente, não permite determinar a gravidade do dano nem a segurança estrutural da edificação.

## 1. Problema e dados
- Tarefa: classificação binária (com/sem fissura) em concreto. Base: *Concrete Crack Images for Classification* (Mendeley), imagens 227×227.
- Amostra: 6000 imagens balanceadas (de 20000 com fissura e 20000 sem fissura disponíveis).
- Divisão sem vazamento: grupos de origem estimados por K-Means sobre características da CNN (200 grupos) e `GroupShuffleSplit`. Treino/validação/teste: 4141/1086/773 imagens, nenhum grupo repetido entre conjuntos. Limitação: os nomes dos arquivos não identificam a foto original, então o agrupamento é aproximado.
- Pré-processamento: redimensionamento 224×224, normalização ImageNet; aumento (giros, brilho/contraste/saturação) só no treino.

## 2. Modelo
- MobileNetV3-Small pré-treinada (ImageNet) com saída de 1 neurônio. Treino em duas etapas: cabeça (3 épocas, backbone congelado) e ajuste fino (5 épocas, taxa 10× menor). Melhor época pela F1 de validação: 7. Tempo de treino: 192 s em Tesla T4.

## 3. Métricas no conjunto de teste
| Limiar | Acurácia | Precisão | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| 0,50 | 0.9974 | 1.0000 | 0.9957 | 0.9979 | 467 | 0 | 2 | 304 |
| 0.31 (escolhido) | 0.9987 | 1.0000 | 0.9979 | 0.9989 | 468 | 0 | 1 | 304 |

AUC no teste: 1.0000. Figuras: `fig03_matriz_confusao_pr.png`.

## 4. Desempenho
- Equipamento: CPU Intel(R) Xeon(R) CPU @ 2.00GHz (2 núcleos lógicos, 1 threads); GPU Tesla T4. Resolução de entrada do modelo: 224×224.
- CPU: latência média 10.24 ms/imagem (p95 11.81 ms); 190.8 imagens/s com lote 32.
- GPU: latência média 5.82 ms/imagem (p95 7.29 ms); 3187.3 imagens/s com lote 32.
- Ponta a ponta (arquivo → resultado, 1 imagem por vez): 8.6 ms/imagem.
- Vídeo de demonstração (1134x680, 15 blocos de 227 px por quadro, cuda): 60.0 FPS no pipeline completo e 9.6 ms de inferência por quadro.

### Comparação de arquiteturas

       arquitetura dispositivo  params_M  lat_media_ms  lat_p95_ms  imgs_por_s_lote32  F1_teste
mobilenet_v3_small         cpu      1.52         10.24       11.81              190.8    0.9989
mobilenet_v3_small        cuda      1.52          5.82        7.29             3187.3    0.9989
mobilenet_v3_large         cpu      4.20         27.29       35.65               35.1       NaN
mobilenet_v3_large        cuda      4.20          7.51        9.48             1398.9       NaN
          resnet18         cpu     11.18         74.95      199.18               18.7    0.9979
          resnet18        cuda     11.18          3.80        4.81             1079.1    0.9979

## 5. Justificativa de arquitetura, pré-processamento e limiar
- Arquitetura: modelo leve pré-treinado, adequado a monitoramento contínuo em equipamento modesto; tarefa binária de padrão visual marcante não exige rede grande. A comparação acima registra o ganho (ou a falta de ganho) de qualidade de uma rede maior frente ao custo de latência.
- Limiar 0.31: escolhido na validação por F-beta (beta = 2) para privilegiar recall, pois perder uma fissura na triagem custa mais que um alarme extra. O app permite ajustá-lo.
- Varredura em blocos de 227 px no vídeo: preserva a escala de treino; reduzir o quadro inteiro apagaria fissuras finas.

## 6. Análise de erros
No teste houve 0 falsos positivos e 1 falsos negativos. Médias por tipo de resultado (brilho 0–255, contraste = desvio do cinza, textura = variância do Laplaciano):

               resultado   n  brilho_medio  contraste_medio  textura_medio  prob_media
    FN (fissura perdida)   1    178.529999            14.89      41.470001        0.22
TN (acerto: sem fissura) 304    181.309998             9.12      21.940001        0.00
    TP (acerto: fissura) 468    161.899994            29.83     120.510002        0.99

> **Complete antes de entregar:** descreva o que as figuras `fig_fp*.png` e `fig_fn*.png` (com Grad-CAM) mostram — sombras, juntas, manchas, texturas rugosas, fissuras muito finas, baixa luminosidade ou obstáculos — e relacione com a tabela acima.

## 7. Limitações
- Base de um único campus; generalização para outras cidades, câmeras e iluminações não medida.
- Divisão por grupos estimados (aproximada).
- Localização por blocos (aproximação), não detecção por caixas delimitadoras.
- O sistema não estima largura, extensão ou gravidade; apenas sinaliza locais para inspeção.