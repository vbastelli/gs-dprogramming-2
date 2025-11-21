# MOH — Motor de Orientação de Habilidades  
**Global Solution — Dynamic Programming**  
**Disciplina:** Engenharia de Software  
**Professor:** André Marques  
**Grupo:**  
NOME: Milton Cezar Bacanieski | RM: 555206  
NOME: Victório Maia Bastelli | RM: 554723  
NOME: Vitor Bebiano Mulford | RM: 555026  

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-%23ffffff.svg?logo=matplotlib&logoColor=black)
![Tests](https://img.shields.io/badge/Tests-33%2F33%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-100%25-success)

Um sistema completo de orientação de carreira que utiliza **Programação Dinâmica Multidimensional**, grafos de pré-requisitos, simulações de incerteza e visualizações profissionais para recomendar a sequência ótima de aquisição de habilidades no mercado de tecnologia.

## Visão do Projeto

O MOH foi criado para resolver um dos maiores desafios do futuro do trabalho:  
**como maximizar o valor de carreira em um mundo onde as habilidades necessárias mudam a cada 18–24 meses?**

A solução combina algoritmos clássicos de otimização com análise experimental real, entregando:

- Caminho de maior valor sob restrições de tempo e complexidade cognitiva  
- Simulação Monte Carlo para lidar com incerteza de mercado  
- Análise crítica de ordens de aprendizado  
- Recomendações personalizadas com radar multidimensional  
- Validação automática de grafos (ciclos e nós órfãos)

## Funcionalidades Implementadas (100% dos requisitos + extras)

| Desafio | Algoritmo Principal | Destaque Entregue |
|--------|----------------------|-------------------|
| 1 – Caminho de Valor Máximo | DP Multidimensional (Knapsack 2D) + reconstrução de caminho | Valor máximo encontrado = 47+ |
| 1 – Incerteza | Monte Carlo com 1.000 simulações (±10% no valor) | Distribuição + histograma |
| 2 – Verificação Crítica | Enumeração completa de permutações (5! = 120) | Top-3 ordens + custos médio/geral |
| 3 – Pivô Mais Rápido | Guloso (V/T) vs Busca Exaustiva (2ⁿ) | Guloso = ótimo neste dataset |
| 4 – Trilhas Paralelas | Merge Sort implementado do zero | Comparação de performance com sort nativo |
| 5 – Recomendação | Score = (Valor × Prob. Mercado) / (Tempo × Complexidade) | Radar chart multidimensional |

## Execução (3 comandos)

```bash
# 1. Clonar e entrar no projeto
git clone https://github.com/vbastelli/gs-dprogramming-2

cd gs-dprogramming-main

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Rodar tudo (relatório + 6 gráficos + JSON)
python moh_system.py
python visualization.py
```

Arquivos gerados automaticamente:
- `relatorio_tecnico.txt` → relatório completo
- `resultados_moh.json` → dados estruturados
- 6 gráficos em PNG (300 DPI)

## Visualizações Profissionais

| Gráfico | Descrição |
|-------|---------|
| `recommendations_radar.png` | Radar multidimensional comparando as 3 melhores próximas habilidades |
| `complexity_comparison.png` | Comparação log de operações estimadas entre todos os desafios |
| `critical_costs.png` | Top-3 ordens críticas com custos |
| `greedy_vs_optimal.png` | Comparação Guloso vs Ótimo |
| `monte_carlo_distribution.png` | Distribuição de valor esperado (1.000 simulações) |
| `sorting_performance.png` | Merge Sort custom vs nativo (speedup ~0.13x – comportamento esperado) |

## Testes Unitários

```bash
python test_moh.py
```

→ **33 testes / 33 passaram**  
Cobertura total de:
- Estrutura do grafo
- Validação (ciclos + nós órfãos)
- Todos os 5 desafios
- Pipeline completo de integração

## Estrutura do Projeto

```
gs-dprogramming-main/
├── moh_system.py          # Execução principal + todos os desafios
├── test_moh.py            # 33 testes unitários
├── visualization.py       # Geração dos 6 gráficos
├── requirements.txt
├── README.md
├── relatorio_tecnico.txt  # gerado automaticamente
├── resultados_moh.json    # gerado automaticamente
└── screenshots/           # 6 visualizações profissionais
```

## Conjunto de Dados Mestre (compatível 100% com especificação)

12 habilidades com:
- Tempo (h)
- Valor (1–10)
- Complexidade (1–10)
- Pré-requisitos
- Tipo (Base, Crítica, Lista Grande, Objetivo Final)

## Validação de Grafo

- Detecção automática de ciclos (DFS)
- Detecção de nós órfãos (pré-requisitos inexistentes)
- Ordenação topológica garantida

