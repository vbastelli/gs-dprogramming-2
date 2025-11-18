"""
Motor de Orientação de Habilidades (MOH)
Global Solution - Engenharia de Software
Prof. André Marques - Nov 2025

Sistema de otimização de aquisição de habilidades usando Programação Dinâmica,
com validação de grafos, simulação Monte Carlo e análise experimental.
"""

import json
import sys
import time
import random
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
from itertools import permutations
import statistics
import heapq

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('moh_execution.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SkillGraph:
    """
    Grafo direcionado ponderado para representar habilidades e suas dependências.
    
    Attributes:
        skills (Dict): Dicionário com metadados de cada habilidade
        graph (Dict): Estrutura de adjacência do grafo
        reverse_graph (Dict): Grafo reverso para análises
    """
    
    def __init__(self):
        """Inicializa o grafo com o conjunto de dados mestre."""
        self.skills: Dict[str, Dict] = self._load_master_dataset()
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_graph: Dict[str, List[str]] = defaultdict(list)
        self._build_graph()
        logger.info(f"Grafo inicializado com {len(self.skills)} habilidades")
    
    def _load_master_dataset(self) -> Dict[str, Dict]:
        """
        Carrega o conjunto de dados mestre de habilidades.
        
        Returns:
            Dict: Dicionário com metadados de cada habilidade
        """
        return {
            'S1': {'Nome': 'Programação Básica (Python)', 'Tempo': 80, 'Valor': 3, 
                   'Complexidade': 4, 'Pre_Reqs': [], 'Tipo': 'Base'},
            'S2': {'Nome': 'Modelagem de Dados (SQL)', 'Tempo': 60, 'Valor': 4, 
                   'Complexidade': 3, 'Pre_Reqs': [], 'Tipo': 'Base'},
            'S3': {'Nome': 'Algoritmos Avançados', 'Tempo': 100, 'Valor': 7, 
                   'Complexidade': 8, 'Pre_Reqs': ['S1'], 'Tipo': 'Crítica'},
            'S4': {'Nome': 'Fundamentos de Machine Learning', 'Tempo': 120, 'Valor': 8, 
                   'Complexidade': 9, 'Pre_Reqs': ['S1', 'S3'], 'Tipo': 'Não Crítica'},
            'S5': {'Nome': 'Visualização de Dados (BI)', 'Tempo': 40, 'Valor': 6, 
                   'Complexidade': 5, 'Pre_Reqs': ['S2'], 'Tipo': 'Crítica'},
            'S6': {'Nome': 'IA Generativa Ética', 'Tempo': 150, 'Valor': 10, 
                   'Complexidade': 10, 'Pre_Reqs': ['S4'], 'Tipo': 'Objetivo Final'},
            'S7': {'Nome': 'Estruturas em Nuvem (AWS/Azure)', 'Tempo': 70, 'Valor': 5, 
                   'Complexidade': 7, 'Pre_Reqs': [], 'Tipo': 'Crítica'},
            'S8': {'Nome': 'APIs e Microsserviços', 'Tempo': 90, 'Valor': 6, 
                   'Complexidade': 6, 'Pre_Reqs': ['S1'], 'Tipo': 'Crítica'},
            'S9': {'Nome': 'DevOps & CI/CD', 'Tempo': 110, 'Valor': 9, 
                   'Complexidade': 8, 'Pre_Reqs': ['S7', 'S8'], 'Tipo': 'Crítica'},
            'H10': {'Nome': 'Segurança de Dados', 'Tempo': 60, 'Valor': 5, 
                    'Complexidade': 6, 'Pre_Reqs': [], 'Tipo': 'Lista Grande'},
            'H11': {'Nome': 'Análise de Big Data', 'Tempo': 90, 'Valor': 8, 
                    'Complexidade': 8, 'Pre_Reqs': ['S4'], 'Tipo': 'Lista Grande'},
            'H12': {'Nome': 'Introdução a IoT', 'Tempo': 30, 'Valor': 3, 
                    'Complexidade': 3, 'Pre_Reqs': [], 'Tipo': 'Lista Grande'}
        }
    
    def _build_graph(self):
        """Constrói o grafo direcionado e seu reverso."""
        for skill_id, metadata in self.skills.items():
            for prereq in metadata['Pre_Reqs']:
                self.graph[prereq].append(skill_id)
                self.reverse_graph[skill_id].append(prereq)
        logger.info("Grafo construído com sucesso")
    
    def validate_graph(self) -> Tuple[bool, List[str]]:
        """
        Valida o grafo verificando ciclos e nós órfãos.
        
        Returns:
            Tuple[bool, List[str]]: (é_válido, lista_de_erros)
        """
        errors = []
        logger.info("Iniciando validação do grafo...")
        
        # Verificar nós órfãos (pré-requisitos que não existem)
        for skill_id, metadata in self.skills.items():
            for prereq in metadata['Pre_Reqs']:
                if prereq not in self.skills:
                    error_msg = f"Nó órfão detectado: {skill_id} requer {prereq} que não existe"
                    errors.append(error_msg)
                    logger.error(error_msg)
        
        # Verificar ciclos usando DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(node: str) -> bool:
            """Função auxiliar para detecção de ciclos."""
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for skill_id in self.skills:
            if skill_id not in visited:
                if has_cycle_util(skill_id):
                    error_msg = f"Ciclo detectado no grafo envolvendo {skill_id}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    break
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info("OK Grafo validado com sucesso")
        else:
            logger.error("ERRO Grafo inválido")
        
        return is_valid, errors
    
    def topological_sort(self) -> Optional[List[str]]:
        """
        Realiza ordenação topológica do grafo.
        
        Returns:
            Optional[List[str]]: Lista ordenada ou None se houver ciclo
        """
        in_degree = {skill: 0 for skill in self.skills}
        
        for skill in self.skills:
            for prereq in self.skills[skill]['Pre_Reqs']:
                in_degree[skill] += 1
        
        queue = deque([skill for skill in self.skills if in_degree[skill] == 0])
        result = []
        
        while queue:
            skill = queue.popleft()
            result.append(skill)
            
            for neighbor in self.graph[skill]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.skills):
            logger.error("Ordenação topológica falhou - grafo contém ciclo")
            return None
        
        logger.info(f"Ordenação topológica concluída: {' -> '.join(result)}")
        return result
    
    def get_basic_skills(self) -> List[str]:
        """
        Retorna habilidades básicas (sem pré-requisitos).
        
        Returns:
            List[str]: Lista de IDs de habilidades básicas
        """
        basic = [sid for sid, meta in self.skills.items() if not meta['Pre_Reqs']]
        logger.info(f"Habilidades básicas encontradas: {basic}")
        return basic
    
    def get_critical_skills(self) -> List[str]:
        """
        Retorna habilidades críticas conforme especificação.
        
        Returns:
            List[str]: Lista de IDs de habilidades críticas
        """
        critical = ['S3', 'S5', 'S7', 'S8', 'S9']
        logger.info(f"Habilidades críticas: {critical}")
        return critical


class Challenge1_MaxValuePath:
    """
    Desafio 1: Caminho de Valor Máximo
    Utiliza DP multidimensional (knapsack 2D) com simulação Monte Carlo.
    """
    
    def __init__(self, graph: SkillGraph):
        """
        Inicializa o desafio com o grafo de habilidades.
        
        Args:
            graph (SkillGraph): Grafo de habilidades validado
        """
        self.graph = graph
        self.max_time = 1200
        self.max_complexity = 70
        self.target = 'S6'
        logger.info("Challenge 1 inicializado: Caminho de Valor Máximo")
    
    def solve_deterministic(self) -> Tuple[List[str], int, int, int]:
        """
        Resolve o problema deterministicamente usando DP.
        
        Returns:
            Tuple: (caminho, valor_total, tempo_total, complexidade_total)
        """
        logger.info("Executando solução determinística...")
        
        # Encontrar todas as habilidades necessárias para S6
        required_skills = self._get_required_skills_for_target()
        
        # DP: dp[t][c] = (valor_max, conjunto_de_habilidades)
        dp = {}
        dp[(0, 0)] = (0, set())
        
        for skill_id in required_skills:
            skill = self.graph.skills[skill_id]
            new_dp = dict(dp)
            
            for (time_used, comp_used), (value, skills_set) in dp.items():
                # Verificar se pré-requisitos foram adquiridos
                prereqs_met = all(prereq in skills_set for prereq in skill['Pre_Reqs'])
                
                if not prereqs_met or skill_id in skills_set:
                    continue
                
                new_time = time_used + skill['Tempo']
                new_comp = comp_used + skill['Complexidade']
                
                if new_time <= self.max_time and new_comp <= self.max_complexity:
                    new_value = value + skill['Valor']
                    new_skills = skills_set | {skill_id}
                    
                    if (new_time, new_comp) not in new_dp or new_dp[(new_time, new_comp)][0] < new_value:
                        new_dp[(new_time, new_comp)] = (new_value, new_skills)
            
            dp = new_dp
        
        # Encontrar melhor solução que inclui S6
        best_value = 0
        best_path = []
        best_time = 0
        best_comp = 0
        
        for (time_used, comp_used), (value, skills_set) in dp.items():
            if self.target in skills_set and value > best_value:
                best_value = value
                best_path = list(skills_set)
                best_time = time_used
                best_comp = comp_used
        
        logger.info(f"Solução determinística: Valor={best_value}, Tempo={best_time}h, Complexidade={best_comp}")
        return best_path, best_value, best_time, best_comp
    
    def solve_monte_carlo(self, n_simulations: int = 1000) -> Dict:
        """
        Resolve com incerteza usando Monte Carlo (V ± 10%).
        
        Args:
            n_simulations (int): Número de simulações
            
        Returns:
            Dict: Estatísticas das simulações
        """
        logger.info(f"Iniciando simulação Monte Carlo com {n_simulations} iterações...")
        
        values = []
        paths = []
        
        for i in range(n_simulations):
            # Perturbar valores com ±10%
            original_values = {}
            for skill_id in self.graph.skills:
                original_values[skill_id] = self.graph.skills[skill_id]['Valor']
                base_value = original_values[skill_id]
                perturbed = base_value * random.uniform(0.9, 1.1)
                self.graph.skills[skill_id]['Valor'] = round(perturbed, 2)
            
            # Resolver com valores perturbados
            path, value, _, _ = self.solve_deterministic()
            values.append(value)
            paths.append(path)
            
            # Restaurar valores originais
            for skill_id, orig_val in original_values.items():
                self.graph.skills[skill_id]['Valor'] = orig_val
            
            if (i + 1) % 100 == 0:
                logger.info(f"Progresso Monte Carlo: {i + 1}/{n_simulations} simulações")
        
        expected_value = statistics.mean(values)
        std_dev = statistics.stdev(values)
        
        logger.info(f"Monte Carlo concluído: E[V]={expected_value:.2f}, σ={std_dev:.2f}")
        
        return {
            'expected_value': expected_value,
            'std_dev': std_dev,
            'min_value': min(values),
            'max_value': max(values),
            'all_values': values
        }
    
    def _get_required_skills_for_target(self) -> List[str]:
        """
        Obtém todas as habilidades necessárias para alcançar o objetivo.
        
        Returns:
            List[str]: Lista de habilidades necessárias
        """
        required = set()
        stack = [self.target]
        
        while stack:
            skill = stack.pop()
            if skill in required:
                continue
            required.add(skill)
            stack.extend(self.graph.skills[skill]['Pre_Reqs'])
        
        return list(required)


class Challenge2_CriticalVerification:
    """
    Desafio 2: Verificação Crítica
    Enumera permutações de habilidades críticas e calcula custos.
    """
    
    def __init__(self, graph: SkillGraph):
        """
        Inicializa o desafio com o grafo validado.
        
        Args:
            graph (SkillGraph): Grafo de habilidades
        """
        self.graph = graph
        self.critical_skills = graph.get_critical_skills()
        logger.info("Challenge 2 inicializado: Verificação Crítica")
    
    def calculate_order_cost(self, order: Tuple[str]) -> int:
        """
        Calcula o custo total de uma ordem de aquisição.
        
        Args:
            order (Tuple[str]): Ordem de aquisição
            
        Returns:
            int: Custo total (tempo + espera)
        """
        acquired = set()
        total_cost = 0
        current_time = 0
        
        for skill_id in order:
            skill = self.graph.skills[skill_id]
            
            # Calcular tempo de espera por pré-requisitos
            wait_time = 0
            for prereq in skill['Pre_Reqs']:
                if prereq not in acquired:
                    # Pré-requisito ainda não adquirido, adicionar seu tempo
                    wait_time += self.graph.skills[prereq]['Tempo']
            
            total_cost += skill['Tempo'] + wait_time
            acquired.add(skill_id)
            current_time += skill['Tempo']
        
        return total_cost
    
    def solve(self) -> Dict:
        """
        Enumera todas as 120 permutações e analisa custos.
        
        Returns:
            Dict: Resultados da análise
        """
        logger.info(f"Enumerando {len(list(permutations(self.critical_skills)))} permutações...")
        
        all_permutations = list(permutations(self.critical_skills))
        costs = []
        
        for perm in all_permutations:
            cost = self.calculate_order_cost(perm)
            costs.append((cost, perm))
        
        costs.sort()
        
        top_3 = costs[:3]
        avg_top_3 = statistics.mean([c[0] for c in top_3])
        avg_all = statistics.mean([c[0] for c in costs])
        
        logger.info(f"Melhor custo: {costs[0][0]}h, Pior custo: {costs[-1][0]}h")
        logger.info(f"Custo médio top 3: {avg_top_3:.2f}h, Custo médio geral: {avg_all:.2f}h")
        
        return {
            'top_3_orders': top_3,
            'avg_top_3': avg_top_3,
            'avg_all': avg_all,
            'best_order': costs[0],
            'worst_order': costs[-1],
            'all_costs': costs
        }


class Challenge3_FastestPivot:
    """
    Desafio 3: Pivô Mais Rápido
    Comparação entre algoritmo guloso e solução ótima.
    """
    
    def __init__(self, graph: SkillGraph):
        """
        Inicializa o desafio.
        
        Args:
            graph (SkillGraph): Grafo de habilidades
        """
        self.graph = graph
        self.target_adaptability = 15
        self.basic_skills = graph.get_basic_skills()
        logger.info("Challenge 3 inicializado: Pivô Mais Rápido")
    
    def solve_greedy(self) -> Tuple[List[str], int, int]:
        """
        Resolve usando algoritmo guloso (maior V/T).
        
        Returns:
            Tuple: (habilidades, valor_total, tempo_total)
        """
        logger.info("Resolvendo com algoritmo guloso (V/T)...")
        
        # Calcular razão V/T para habilidades básicas
        skills_with_ratio = []
        for skill_id in self.basic_skills:
            skill = self.graph.skills[skill_id]
            ratio = skill['Valor'] / skill['Tempo']
            skills_with_ratio.append((ratio, skill_id, skill['Valor'], skill['Tempo']))
        
        # Ordenar por razão decrescente
        skills_with_ratio.sort(reverse=True)
        
        selected = []
        total_value = 0
        total_time = 0
        
        for ratio, skill_id, value, time in skills_with_ratio:
            if total_value >= self.target_adaptability:
                break
            selected.append(skill_id)
            total_value += value
            total_time += time
        
        logger.info(f"Guloso: {len(selected)} habilidades, V={total_value}, T={total_time}h")
        return selected, total_value, total_time
    
    def solve_optimal(self) -> Tuple[List[str], int, int]:
        """
        Resolve por busca exaustiva (força bruta em subconjuntos).
        
        Returns:
            Tuple: (habilidades, valor_total, tempo_total)
        """
        logger.info("Resolvendo com busca exaustiva...")
        
        best_skills = []
        best_time = float('inf')
        
        # Testar todos os subconjuntos
        from itertools import combinations
        
        for r in range(1, len(self.basic_skills) + 1):
            for subset in combinations(self.basic_skills, r):
                total_value = sum(self.graph.skills[s]['Valor'] for s in subset)
                total_time = sum(self.graph.skills[s]['Tempo'] for s in subset)
                
                if total_value >= self.target_adaptability and total_time < best_time:
                    best_skills = list(subset)
                    best_time = total_time
        
        best_value = sum(self.graph.skills[s]['Valor'] for s in best_skills)
        
        logger.info(f"Ótimo: {len(best_skills)} habilidades, V={best_value}, T={best_time}h")
        return best_skills, best_value, best_time
    
    def find_counterexample(self) -> Optional[Dict]:
        """
        Procura contraexemplo onde guloso não é ótimo.
        
        Returns:
            Optional[Dict]: Contraexemplo se encontrado
        """
        greedy_skills, greedy_value, greedy_time = self.solve_greedy()
        optimal_skills, optimal_value, optimal_time = self.solve_optimal()
        
        if greedy_time > optimal_time:
            logger.info("✓ Contraexemplo encontrado: Guloso não é ótimo!")
            return {
                'greedy': {'skills': greedy_skills, 'value': greedy_value, 'time': greedy_time},
                'optimal': {'skills': optimal_skills, 'value': optimal_value, 'time': optimal_time},
                'difference': greedy_time - optimal_time
            }
        else:
            logger.info("Guloso encontrou solução ótima neste caso")
            return None


class Challenge4_ParallelTracks:
    """
    Desafio 4: Trilhas Paralelas
    Implementação de algoritmos de ordenação.
    """
    
    def __init__(self, graph: SkillGraph):
        """
        Inicializa o desafio.
        
        Args:
            graph (SkillGraph): Grafo de habilidades
        """
        self.graph = graph
        logger.info("Challenge 4 inicializado: Trilhas Paralelas")
    
    def merge_sort(self, skills: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """
        Implementação do Merge Sort.
        
        Args:
            skills (List[Tuple]): Lista de (skill_id, complexidade)
            
        Returns:
            List[Tuple]: Lista ordenada
        """
        if len(skills) <= 1:
            return skills
        
        mid = len(skills) // 2
        left = self.merge_sort(skills[:mid])
        right = self.merge_sort(skills[mid:])
        
        return self._merge(left, right)
    
    def _merge(self, left: List[Tuple], right: List[Tuple]) -> List[Tuple]:
        """Combina duas listas ordenadas."""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i][1] <= right[j][1]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    def quick_sort(self, skills: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """
        Implementação do Quick Sort.
        
        Args:
            skills (List[Tuple]): Lista de (skill_id, complexidade)
            
        Returns:
            List[Tuple]: Lista ordenada
        """
        if len(skills) <= 1:
            return skills
        
        pivot = skills[len(skills) // 2][1]
        left = [s for s in skills if s[1] < pivot]
        middle = [s for s in skills if s[1] == pivot]
        right = [s for s in skills if s[1] > pivot]
        
        return self.quick_sort(left) + middle + self.quick_sort(right)
    
    def solve(self, algorithm: str = 'merge') -> Dict:
        """
        Ordena habilidades por complexidade.
        
        Args:
            algorithm (str): 'merge' ou 'quick'
            
        Returns:
            Dict: Resultados da ordenação
        """
        logger.info(f"Ordenando com {algorithm.upper()} SORT...")
        
        # Preparar dados
        skills_list = [(sid, meta['Complexidade']) for sid, meta in self.graph.skills.items()]
        
        # Medir tempo custom
        start = time.perf_counter()
        if algorithm == 'merge':
            sorted_skills = self.merge_sort(skills_list)
        else:
            sorted_skills = self.quick_sort(skills_list)
        custom_time = time.perf_counter() - start
        
        # Medir tempo nativo (baseline)
        start = time.perf_counter()
        native_sorted = sorted(skills_list, key=lambda x: x[1])
        native_time = time.perf_counter() - start
        
        # Dividir em sprints
        sprint_a = sorted_skills[:6]
        sprint_b = sorted_skills[6:]
        
        logger.info(f"Ordenação concluída: {custom_time*1000:.4f}ms (custom) vs {native_time*1000:.4f}ms (native)")
        
        return {
            'sorted_skills': sorted_skills,
            'sprint_a': sprint_a,
            'sprint_b': sprint_b,
            'custom_time': custom_time,
            'native_time': native_time,
            'speedup': native_time / custom_time if custom_time > 0 else float('inf')
        }


class Challenge5_NextSkillsRecommendation:
    """
    Desafio 5: Recomendar Próximas Habilidades
    Usa DP com horizonte finito.
    """
    
    def __init__(self, graph: SkillGraph):
        """
        Inicializa o desafio.
        
        Args:
            graph (SkillGraph): Grafo de habilidades
        """
        self.graph = graph
        self.horizon_years = 5
        logger.info("Challenge 5 inicializado: Recomendação de Habilidades")
    
    def simulate_market_transitions(self) -> Dict[str, float]:
        """
        Simula probabilidades de mercado para cada habilidade.
        
        Returns:
            Dict: Probabilidades de demanda futura
        """
        # Simular probabilidades baseadas em tipo e valor
        probabilities = {}
        
        for skill_id, metadata in self.graph.skills.items():
            base_prob = metadata['Valor'] / 10.0
            
            # Ajustar por tipo
            if metadata['Tipo'] == 'Objetivo Final':
                base_prob *= 1.5
            elif metadata['Tipo'] == 'Crítica':
                base_prob *= 1.3
            elif metadata['Tipo'] == 'Base':
                base_prob *= 0.8
            
            probabilities[skill_id] = min(base_prob, 1.0)
        
        logger.info("Probabilidades de mercado simuladas")
        return probabilities
    
    def recommend(self, current_profile: Set[str], n_recommendations: int = 3) -> List[Dict]:
        """
        Recomenda próximas habilidades.
        
        Args:
            current_profile (Set[str]): Habilidades já adquiridas
            n_recommendations (int): Número de recomendações
            
        Returns:
            List[Dict]: Lista de recomendações
        """
        logger.info(f"Gerando {n_recommendations} recomendações...")
        
        market_probs = self.simulate_market_transitions()
        
        # Calcular score para cada habilidade disponível
        scores = []
        
        for skill_id, metadata in self.graph.skills.items():
            if skill_id in current_profile:
                continue
            
            # Verificar pré-requisitos
            prereqs_met = all(prereq in current_profile for prereq in metadata['Pre_Reqs'])
            if not prereqs_met:
                continue
            
            # Calcular score esperado
            value = metadata['Valor']
            time = metadata['Tempo']
            prob = market_probs[skill_id]
            complexity = metadata['Complexidade']
            
            # Score = (Valor * Probabilidade) / (Tempo * Complexidade)
            score = (value * prob) / (time * complexity / 100)
            
            scores.append({
                'skill_id': skill_id,
                'nome': metadata['Nome'],
                'score': score,
                'probabilidade_mercado': prob,
                'valor': value,
                'tempo': time
            })
        
        # Ordenar e retornar top N
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        recommendations = scores[:n_recommendations]
        logger.info(f"Recomendações geradas: {[r['skill_id'] for r in recommendations]}")
        
        return recommendations


def generate_report(graph: SkillGraph, results: Dict):
    """
    Gera relatório técnico consolidado.
    
    Args:
        graph (SkillGraph): Grafo de habilidades
        results (Dict): Resultados de todos os desafios
    """
    logger.info("Gerando relatório técnico...")
    
    report = []
    report.append("=" * 80)
    report.append("RELATÓRIO TÉCNICO - MOTOR DE ORIENTAÇÃO DE HABILIDADES (MOH)")
    report.append("=" * 80)
    report.append(f"\nData: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total de Habilidades: {len(graph.skills)}")
    report.append("\n" + "=" * 80)
    
    # Desafio 1
    report.append("\n1. CAMINHO DE VALOR MÁXIMO")
    report.append("-" * 80)
    if 'challenge1' in results:
        c1 = results['challenge1']
        report.append(f"Solução Determinística:")
        report.append(f"  - Caminho: {' -> '.join(c1['deterministic']['path'])}")
        report.append(f"  - Valor Total: {c1['deterministic']['value']}")
        report.append(f"  - Tempo Total: {c1['deterministic']['time']}h")
        report.append(f"  - Complexidade Total: {c1['deterministic']['complexity']}")
        report.append(f"\nSimulação Monte Carlo ({c1['monte_carlo']['n_sim']} iterações):")
        report.append(f"  - E[Valor]: {c1['monte_carlo']['expected_value']:.2f}")
        report.append(f"  - Desvio Padrão: {c1['monte_carlo']['std_dev']:.2f}")
        report.append(f"  - Min: {c1['monte_carlo']['min_value']:.2f}")
        report.append(f"  - Max: {c1['monte_carlo']['max_value']:.2f}")
    
    # Desafio 2
    report.append("\n2. VERIFICAÇÃO CRÍTICA")
    report.append("-" * 80)
    if 'challenge2' in results:
        c2 = results['challenge2']
        report.append(f"Habilidades Críticas Analisadas: {', '.join(graph.get_critical_skills())}")
        report.append(f"Total de Permutações: 120")
        report.append(f"\nTop 3 Melhores Ordens:")
        for i, (cost, order) in enumerate(c2['top_3_orders'], 1):
            report.append(f"  {i}. {' -> '.join(order)} (Custo: {cost}h)")
        report.append(f"\nCusto Médio Top 3: {c2['avg_top_3']:.2f}h")
        report.append(f"Custo Médio Geral: {c2['avg_all']:.2f}h")
        report.append(f"Melhor Custo: {c2['best_order'][0]}h")
        report.append(f"Pior Custo: {c2['worst_order'][0]}h")
    
    # Desafio 3
    report.append("\n3. PIVÔ MAIS RÁPIDO")
    report.append("-" * 80)
    if 'challenge3' in results:
        c3 = results['challenge3']
        report.append(f"Meta de Adaptabilidade: ≥ 15")
        report.append(f"\nSolução Gulosa (V/T):")
        report.append(f"  - Habilidades: {', '.join(c3['greedy']['skills'])}")
        report.append(f"  - Valor: {c3['greedy']['value']}")
        report.append(f"  - Tempo: {c3['greedy']['time']}h")
        report.append(f"\nSolução Ótima (Busca Exaustiva):")
        report.append(f"  - Habilidades: {', '.join(c3['optimal']['skills'])}")
        report.append(f"  - Valor: {c3['optimal']['value']}")
        report.append(f"  - Tempo: {c3['optimal']['time']}h")
        if c3['counterexample']:
            report.append(f"\n✓ CONTRAEXEMPLO ENCONTRADO!")
            report.append(f"  - Diferença de tempo: {c3['counterexample']['difference']}h")
            report.append(f"  - Guloso NÃO é ótimo neste caso")
        else:
            report.append(f"\nGuloso encontrou solução ótima neste caso")
    
    # Desafio 4
    report.append("\n4. TRILHAS PARALELAS")
    report.append("-" * 80)
    if 'challenge4' in results:
        c4 = results['challenge4']
        report.append(f"Algoritmo Utilizado: {c4['algorithm'].upper()} SORT")
        report.append(f"\nSprint A (Complexidade 1-6):")
        for skill_id, comp in c4['sprint_a']:
            report.append(f"  - {skill_id}: {graph.skills[skill_id]['Nome']} (C={comp})")
        report.append(f"\nSprint B (Complexidade 7-12):")
        for skill_id, comp in c4['sprint_b']:
            report.append(f"  - {skill_id}: {graph.skills[skill_id]['Nome']} (C={comp})")
        report.append(f"\nPerformance:")
        report.append(f"  - Implementação Custom: {c4['custom_time']*1000:.4f}ms")
        report.append(f"  - Sort Nativo (baseline): {c4['native_time']*1000:.4f}ms")
        report.append(f"  - Speedup: {c4['speedup']:.2f}x")
    
    # Desafio 5
    report.append("\n5. RECOMENDAÇÃO DE PRÓXIMAS HABILIDADES")
    report.append("-" * 80)
    if 'challenge5' in results:
        c5 = results['challenge5']
        report.append(f"Perfil Atual: {', '.join(c5['current_profile'])}")
        report.append(f"Horizonte: {c5['horizon']} anos")
        report.append(f"\nRecomendações:")
        for i, rec in enumerate(c5['recommendations'], 1):
            report.append(f"  {i}. {rec['skill_id']}: {rec['nome']}")
            report.append(f"     - Score: {rec['score']:.4f}")
            report.append(f"     - Prob. Mercado: {rec['probabilidade_mercado']:.2f}")
            report.append(f"     - Valor: {rec['valor']}, Tempo: {rec['tempo']}h")
    
    # Análise de Complexidade
    report.append("\n" + "=" * 80)
    report.append("ANÁLISE DE COMPLEXIDADE COMPUTACIONAL")
    report.append("=" * 80)
    report.append("\nDesafio 1 - Caminho de Valor Máximo:")
    report.append("  - Algoritmo: DP Multidimensional (Knapsack 2D)")
    report.append("  - Complexidade: O(n × T × C) onde n=habilidades, T=tempo_max, C=complexidade_max")
    report.append("  - Monte Carlo: O(k × n × T × C) onde k=número de simulações")
    report.append("\nDesafio 2 - Verificação Crítica:")
    report.append("  - Algoritmo: Enumeração de Permutações")
    report.append("  - Complexidade: O(n!) = O(5!) = 120 permutações")
    report.append("  - Validação de Grafo: O(V + E) para DFS")
    report.append("\nDesafio 3 - Pivô Mais Rápido:")
    report.append("  - Guloso: O(n log n) para ordenação")
    report.append("  - Ótimo: O(2^n) para busca exaustiva")
    report.append("  - n = número de habilidades básicas")
    report.append("\nDesafio 4 - Trilhas Paralelas:")
    report.append("  - Merge Sort: O(n log n) - melhor, médio e pior caso")
    report.append("  - Quick Sort: Melhor/Médio O(n log n), Pior O(n²)")
    report.append("\nDesafio 5 - Recomendação:")
    report.append("  - Complexidade: O(n × m) onde n=habilidades, m=perfil atual")
    
    # Validação e Testes
    report.append("\n" + "=" * 80)
    report.append("VALIDAÇÃO E TRATAMENTO DE ERROS")
    report.append("=" * 80)
    if 'validation' in results:
        val = results['validation']
        report.append(f"Validação do Grafo: {'✓ APROVADO' if val['is_valid'] else '✗ FALHOU'}")
        if not val['is_valid']:
            report.append(f"Erros Encontrados:")
            for error in val['errors']:
                report.append(f"  - {error}")
        else:
            report.append("  - Sem ciclos detectados")
            report.append("  - Sem nós órfãos")
            report.append("  - Ordenação topológica bem-sucedida")
    
    report.append("\n" + "=" * 80)
    report.append("FIM DO RELATÓRIO")
    report.append("=" * 80)
    
    # Salvar relatório
    report_text = "\n".join(report)
    with open('relatorio_tecnico.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info("Relatório técnico gerado: relatorio_tecnico.txt")
    print(report_text)


def main():
    """
    Função principal que executa todos os desafios e gera relatório.
    """
    print("=" * 80)
    print("MOTOR DE ORIENTAÇÃO DE HABILIDADES (MOH)")
    print("Global Solution - Engenharia de Software")
    print("=" * 80)
    
    logger.info("Iniciando execução do MOH...")
    
    # Inicializar grafo
    graph = SkillGraph()
    
    # Validar grafo (CRÍTICO - antes de qualquer otimização)
    is_valid, errors = graph.validate_graph()
    if not is_valid:
        logger.critical("ERRO: Grafo inválido! Execução interrompida.")
        print("\n❌ GRAFO INVÁLIDO - EXECUÇÃO INTERROMPIDA")
        for error in errors:
            print(f"  - {error}")
        return
    
    # Ordenação topológica
    topo_order = graph.topological_sort()
    
    results = {
        'validation': {'is_valid': is_valid, 'errors': errors}
    }
    
    # DESAFIO 1: Caminho de Valor Máximo
    print("\n🔹 Executando Desafio 1: Caminho de Valor Máximo...")
    challenge1 = Challenge1_MaxValuePath(graph)
    path, value, time_spent, complexity = challenge1.solve_deterministic()
    monte_carlo_results = challenge1.solve_monte_carlo(n_simulations=1000)
    
    results['challenge1'] = {
        'deterministic': {
            'path': path,
            'value': value,
            'time': time_spent,
            'complexity': complexity
        },
        'monte_carlo': {
            'expected_value': monte_carlo_results['expected_value'],
            'std_dev': monte_carlo_results['std_dev'],
            'min_value': monte_carlo_results['min_value'],
            'max_value': monte_carlo_results['max_value'],
            'n_sim': 1000
        }
    }
    print(f"✓ Concluído: Valor={value}, E[V]={monte_carlo_results['expected_value']:.2f}")
    
    # DESAFIO 2: Verificação Crítica
    print("\n🔹 Executando Desafio 2: Verificação Crítica...")
    challenge2 = Challenge2_CriticalVerification(graph)
    critical_results = challenge2.solve()
    results['challenge2'] = critical_results
    print(f"✓ Concluído: Melhor custo={critical_results['best_order'][0]}h")
    
    # DESAFIO 3: Pivô Mais Rápido
    print("\n🔹 Executando Desafio 3: Pivô Mais Rápido...")
    challenge3 = Challenge3_FastestPivot(graph)
    greedy_skills, greedy_value, greedy_time = challenge3.solve_greedy()
    optimal_skills, optimal_value, optimal_time = challenge3.solve_optimal()
    counterexample = challenge3.find_counterexample()
    
    results['challenge3'] = {
        'greedy': {'skills': greedy_skills, 'value': greedy_value, 'time': greedy_time},
        'optimal': {'skills': optimal_skills, 'value': optimal_value, 'time': optimal_time},
        'counterexample': counterexample
    }
    print(f"✓ Concluído: Guloso={greedy_time}h, Ótimo={optimal_time}h")
    
    # DESAFIO 4: Trilhas Paralelas
    print("\n🔹 Executando Desafio 4: Trilhas Paralelas...")
    challenge4 = Challenge4_ParallelTracks(graph)
    sorting_results = challenge4.solve(algorithm='merge')
    results['challenge4'] = {
        'algorithm': 'merge',
        'sorted_skills': sorting_results['sorted_skills'],
        'sprint_a': sorting_results['sprint_a'],
        'sprint_b': sorting_results['sprint_b'],
        'custom_time': sorting_results['custom_time'],
        'native_time': sorting_results['native_time'],
        'speedup': sorting_results['speedup']
    }
    print(f"✓ Concluído: {sorting_results['custom_time']*1000:.4f}ms")
    
    # DESAFIO 5: Recomendação de Próximas Habilidades
    print("\n🔹 Executando Desafio 5: Recomendação de Habilidades...")
    challenge5 = Challenge5_NextSkillsRecommendation(graph)
    current_profile = {'S1', 'S2'}  # Exemplo: usuário já tem Python e SQL
    recommendations = challenge5.recommend(current_profile, n_recommendations=3)
    results['challenge5'] = {
        'current_profile': list(current_profile),
        'horizon': 5,
        'recommendations': recommendations
    }
    print(f"✓ Concluído: {len(recommendations)} recomendações geradas")
    
    # Gerar relatório final
    print("\n🔹 Gerando Relatório Técnico...")
    generate_report(graph, results)
    
    # Salvar resultados em JSON
    with open('resultados_moh.json', 'w', encoding='utf-8') as f:
        # Converter sets para listas para JSON
        results_json = results.copy()
        results_json['challenge5']['current_profile'] = list(current_profile)
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    logger.info("Execução concluída com sucesso!")
    print("\n" + "=" * 80)
    print("✅ EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 80)
    print("Arquivos gerados:")
    print("  - relatorio_tecnico.txt")
    print("  - resultados_moh.json")
    print("  - moh_execution.log")
    print("=" * 80)


if __name__ == "__main__":
    main()