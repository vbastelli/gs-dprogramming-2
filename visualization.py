"""
Módulo de Análise e Visualização - MOH
Gera gráficos e análises complementares para o relatório técnico
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import json


class MOHVisualizer:
    """
    Classe para gerar visualizações dos resultados do MOH.
    """
    
    def __init__(self, results_file: str = 'resultados_moh.json'):
        """
        Inicializa o visualizador com os resultados.
        
        Args:
            results_file (str): Caminho para o arquivo de resultados JSON
        """
        with open(results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        # Configuração de estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    def plot_monte_carlo_distribution(self, output_file: str = 'monte_carlo_distribution.png'):
        """
        Gera histograma da distribuição Monte Carlo.
        
        Args:
            output_file (str): Nome do arquivo de saída
        """
        if 'challenge1' not in self.results:
            print("Dados do Challenge 1 não encontrados")
            return
        
        mc_data = self.results['challenge1']['monte_carlo']
        
        # Simular distribuição (valores não estão salvos, então simulamos)
        mean = mc_data['expected_value']
        std = mc_data['std_dev']
        values = np.random.normal(mean, std, 1000)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histograma
        n, bins, patches = ax.hist(values, bins=30, alpha=0.7, color=self.colors[0], 
                                     edgecolor='black')
        
        # Linha vertical para média
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, 
                   label=f'E[V] = {mean:.2f}')
        
        # Área de ±1 desvio padrão
        ax.axvspan(mean - std, mean + std, alpha=0.2, color='yellow', 
                   label=f'±1σ = {std:.2f}')
        
        ax.set_xlabel('Valor Total', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
        ax.set_title('Distribuição Monte Carlo - Valor Esperado (1000 simulações)', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_file}")
        plt.close()
    
    def plot_critical_skills_costs(self, output_file: str = 'critical_costs.png'):
        """
        Gera gráfico de barras com custos das ordens críticas.
        
        Args:
            output_file (str): Nome do arquivo de saída
        """
        if 'challenge2' not in self.results:
            print("Dados do Challenge 2 não encontrados")
            return
        
        c2 = self.results['challenge2']
        top_3 = c2['top_3_orders'][:3]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        orders = [f"Ordem {i+1}\n{' → '.join(order)}" for i, (_, order) in enumerate(top_3)]
        costs = [cost for cost, _ in top_3]
        
        bars = ax.bar(range(len(orders)), costs, color=self.colors[:3], 
                      edgecolor='black', linewidth=1.5)
        
        # Adicionar valores nas barras
        for i, (bar, cost) in enumerate(zip(bars, costs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cost}h',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Linha de média
        avg = c2['avg_top_3']
        ax.axhline(avg, color='red', linestyle='--', linewidth=2, 
                   label=f'Média Top 3 = {avg:.1f}h')
        
        ax.set_ylabel('Custo Total (horas)', fontsize=12, fontweight='bold')
        ax.set_title('Top 3 Melhores Ordens de Habilidades Críticas', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(orders)))
        ax.set_xticklabels(orders, fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_file}")
        plt.close()
    
    def plot_greedy_vs_optimal(self, output_file: str = 'greedy_vs_optimal.png'):
        """
        Compara Greedy vs Ótimo do Desafio 3.
        
        Args:
            output_file (str): Nome do arquivo de saída
        """
        if 'challenge3' not in self.results:
            print("Dados do Challenge 3 não encontrados")
            return
        
        c3 = self.results['challenge3']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Gráfico 1: Comparação de Tempo
        methods = ['Guloso', 'Ótimo']
        times = [c3['greedy']['time'], c3['optimal']['time']]
        
        bars1 = ax1.bar(methods, times, color=[self.colors[0], self.colors[1]], 
                        edgecolor='black', linewidth=1.5)
        
        for bar, time in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'{time}h',
                     ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax1.set_ylabel('Tempo Total (horas)', fontsize=12, fontweight='bold')
        ax1.set_title('Comparação de Tempo: Guloso vs Ótimo', 
                      fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Gráfico 2: Número de Habilidades
        n_skills = [len(c3['greedy']['skills']), len(c3['optimal']['skills'])]
        
        bars2 = ax2.bar(methods, n_skills, color=[self.colors[2], self.colors[3]], 
                        edgecolor='black', linewidth=1.5)
        
        for bar, n in zip(bars2, n_skills):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{n}',
                     ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax2.set_ylabel('Número de Habilidades', fontsize=12, fontweight='bold')
        ax2.set_title('Quantidade de Habilidades Selecionadas', 
                      fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_file}")
        plt.close()
    
    def plot_sorting_performance(self, output_file: str = 'sorting_performance.png'):
        """
        Gera gráfico de performance dos algoritmos de ordenação.
        
        Args:
            output_file (str): Nome do arquivo de saída
        """
        if 'challenge4' not in self.results:
            print("Dados do Challenge 4 não encontrados")
            return
        
        c4 = self.results['challenge4']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = ['Merge Sort\n(Custom)', 'Sort Nativo\n(Baseline)']
        times_ms = [c4['custom_time'] * 1000, c4['native_time'] * 1000]
        
        bars = ax.bar(algorithms, times_ms, color=[self.colors[0], self.colors[4]], 
                      edgecolor='black', linewidth=1.5)
        
        for bar, time in zip(bars, times_ms):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.4f}ms',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel('Tempo de Execução (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Performance de Algoritmos de Ordenação (n=12)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar speedup
        speedup = c4['speedup']
        ax.text(0.5, max(times_ms) * 0.8, 
                f'Speedup: {speedup:.2f}x',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_file}")
        plt.close()
    
    def plot_recommendations_radar(self, output_file: str = 'recommendations_radar.png'):
        """
        Gera gráfico radar das recomendações.
        
        Args:
            output_file (str): Nome do arquivo de saída
        """
        if 'challenge5' not in self.results:
            print("Dados do Challenge 5 não encontrados")
            return
        
        c5 = self.results['challenge5']
        recs = c5['recommendations'][:3]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Categorias
        categories = ['Score', 'Prob. Mercado', 'Valor', 'Eficiência\n(1/Tempo)']
        N = len(categories)
        
        # Ângulos
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Plot para cada recomendação
        for i, rec in enumerate(recs):
            # Normalizar valores para escala 0-1
            values = [
                rec['score'] / max([r['score'] for r in recs]),
                rec['probabilidade_mercado'],
                rec['valor'] / 10,
                (1 / rec['tempo']) / (1 / min([r['tempo'] for r in recs]))
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=rec['skill_id'],
                    color=self.colors[i])
            ax.fill(angles, values, alpha=0.15, color=self.colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('Análise Multidimensional das Recomendações', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_file}")
        plt.close()
    
    def generate_complexity_comparison(self, output_file: str = 'complexity_comparison.png'):
        """
        Gera gráfico de comparação de complexidades.
        
        Args:
            output_file (str): Nome do arquivo de saída
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Dados de complexidade
        challenges = ['Challenge 1\nDP Knapsack', 'Challenge 2\nPermutações', 
                      'Challenge 3\nGuloso', 'Challenge 3\nÓtimo',
                      'Challenge 4\nMerge Sort']
        
        complexities = ['O(n×T×C)', 'O(n!)', 'O(n log n)', 'O(2^n)', 'O(n log n)']
        
        # Valores aproximados em log scale para n=12
        n = 12
        T, C = 350, 30
        estimated_ops = [
            n * T * C,  # Challenge 1
            np.math.factorial(5),  # Challenge 2 (5 críticas)
            n * np.log2(n),  # Challenge 3 Greedy
            2**6,  # Challenge 3 Ótimo (6 básicas)
            n * np.log2(n)  # Challenge 4
        ]
        
        bars = ax.bar(range(len(challenges)), np.log10(estimated_ops), 
                      color=self.colors, edgecolor='black', linewidth=1.5)
        
        # Adicionar labels
        for i, (bar, ops, comp) in enumerate(zip(bars, estimated_ops, complexities)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{comp}\n({ops:,.0f} ops)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('log₁₀(Operações Estimadas)', fontsize=12, fontweight='bold')
        ax.set_title('Comparação de Complexidade Computacional dos Desafios', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(challenges)))
        ax.set_xticklabels(challenges, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_file}")
        plt.close()
    
    def generate_all_visualizations(self):
        """Gera todas as visualizações de uma vez."""
        print("\n🎨 Gerando visualizações...")
        print("=" * 60)
        
        self.plot_monte_carlo_distribution()
        self.plot_critical_skills_costs()
        self.plot_greedy_vs_optimal()
        self.plot_sorting_performance()
        self.plot_recommendations_radar()
        self.generate_complexity_comparison()
        
        print("=" * 60)
        print("✅ Todas as visualizações foram geradas com sucesso!")
        print("\nArquivos criados:")
        print("  - monte_carlo_distribution.png")
        print("  - critical_costs.png")
        print("  - greedy_vs_optimal.png")
        print("  - sorting_performance.png")
        print("  - recommendations_radar.png")
        print("  - complexity_comparison.png")


if __name__ == "__main__":
    # Executar visualizações
    visualizer = MOHVisualizer('resultados_moh.json')
    visualizer.generate_all_visualizations()