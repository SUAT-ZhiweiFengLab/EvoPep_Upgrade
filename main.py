import torch
from evopep import EvoPep

def main():
    print("==================================================")
    print("EvoPep Framework: AI-Driven Cyclic Peptide Design")
    print("Core: MPNN (from EpiMII) + CNN + Transformer Attn")
    print("==================================================")
    
    # 起始脚手架 (类似 Exendin-4 和 GLP-1)
    initial_seeds = [
        "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS",
        "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"
    ]
    
    # 实例化并运行框架
    evopep_system = EvoPep(pop_size=40, generations=5, top_k=5)
    best_candidates = evopep_system.run(initial_seeds)
    
    print("\n==================================================")
    print("Evolution Complete. Final Lead Candidates:")
    print("==================================================")
    best = best_candidates[0]
    print(f"Sequence: {best['sequence']}")
    print(f"Multi-objective Fitness Score: {best['fitness']:.4f}")
    print("Predicted pKd Scores across AD targets:")
    for tgt, score in best['scores'].items():
        print(f" - {tgt}: {score:.2f} (Uncertainty: +/- {best['intervals'][tgt]:.2f})")

if __name__ == "__main__":
    main()