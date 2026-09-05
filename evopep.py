import torch
from models import EvoPepHybridGenerator, ConformalQSARPredictor

class EvoPep:
    def __init__(self, pop_size=40, generations=5, top_k=5):
        self.pop_size = pop_size
        self.generations = generations
        self.top_k = top_k
        
        # 初始化升级版生成器 (EpiMII MPNN 继承 + CNN/Attn 优化)
        self.generator = EvoPepHybridGenerator()
        self.generator.load_state_dict(torch.load('pretrained_hybrid.pth'))
        
        self.targets = ['GLP-1R', 'GRIN2A', 'HDAC6', 'CHRNA7', 'SIGMAR1', 'PI3K', 'MEK1']
        self.evaluator = ConformalQSARPredictor(self.targets)
        
    def calculate_fitness(self, evaluation_result):
        scores = evaluation_result['scores']
        intervals = evaluation_result['intervals']
        
        # 多目标打分：以 GLP-1R 为主，兼顾 AD 多靶点，严惩共形预测不确定性
        fitness = scores['GLP-1R'] * 2.0 
        off_target_score = sum([scores[t] for t in self.targets if t != 'GLP-1R']) / (len(self.targets) - 1)
        fitness += off_target_score
        
        avg_uncertainty = sum(intervals.values()) / len(intervals)
        fitness -= (avg_uncertainty * 1.0) 
        
        return fitness

    def run(self, initial_seeds):
        population = initial_seeds
        
        for gen in range(self.generations):
            print(f"--- Generation {gen + 1} ---")
            eval_results = self.evaluator.predict(population)
            
            for res in eval_results:
                res['fitness'] = self.calculate_fitness(res)
                
            eval_results.sort(key=lambda x: x['fitness'], reverse=True)
            
            top_candidates = eval_results[:self.top_k]
            top_sequences = [x['sequence'] for x in top_candidates]
            
            print(f"Top Fitness: {top_candidates[0]['fitness']:.4f} | Seq: {top_sequences[0]}")
            
            if gen == self.generations - 1:
                return top_candidates
                
            # 由神经网络指导下一代种群生成
            population = self.generator.generate_variants(
                seed_sequences=top_sequences, 
                num_variants=max(1, self.pop_size // self.top_k),
                temperature=1.2,
                mutation_rate=0.2
            )
            
        return []