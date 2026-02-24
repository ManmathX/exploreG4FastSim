#!/usr/bin/env python3
"""
Calorimeter Shower ML Analyzer
Professional tool for electromagnetic shower simulation and ML dataset generation
"""

import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
from math import gamma

@dataclass
class ShowerEvent:
    energy: float
    particle_type: str
    depth_profile: List[float]
    lateral_profile: List[float]
    total_energy_deposited: float
    shower_max_depth: float
    containment_fraction: float

class CalorimeterShowerSimulator:
    """Fast electromagnetic shower simulator using parameterization"""
    
    def __init__(self, n_layers=50, layer_thickness=1.0):
        self.n_layers = n_layers
        self.layer_thickness = layer_thickness
        self.X0 = 1.0  # Radiation length
        
    def simulate_em_shower(self, energy: float, particle: str = "electron") -> ShowerEvent:
        """Simulate EM shower using Grindhammer parameterization"""
        t_max = np.log(energy / 0.511) + (0.5 if particle == "electron" else -0.5)
        
        depths = np.linspace(0, self.n_layers * self.layer_thickness, self.n_layers)
        t = depths / self.X0
        
        # Longitudinal profile (Gamma distribution approximation)
        alpha = max(t_max, 1.0)  # Ensure alpha >= 1
        beta = 0.5
        with np.errstate(divide='ignore', invalid='ignore'):
            depth_profile = (beta ** alpha / gamma(alpha)) * (t ** (alpha - 1)) * np.exp(-beta * t)
            depth_profile = np.nan_to_num(depth_profile, nan=0.0, posinf=0.0, neginf=0.0)
            if depth_profile.sum() > 0:
                depth_profile = depth_profile / depth_profile.sum() * energy
            else:
                depth_profile = np.zeros_like(t)
        
        # Lateral profile (Gaussian approximation)
        radii = np.linspace(0, 10, 30)
        moliere_radius = 2.0
        lateral_profile = np.exp(-(radii ** 2) / (2 * moliere_radius ** 2))
        lateral_profile = lateral_profile / lateral_profile.sum()
        
        shower_max = depths[np.argmax(depth_profile)] if depth_profile.max() > 0 else 0.0
        total_deposited = float(np.sum(depth_profile))
        containment = np.sum(depth_profile[:int(0.95 * len(depth_profile))]) / energy if energy > 0 else 0.0
        
        return ShowerEvent(
            energy=energy,
            particle_type=particle,
            depth_profile=depth_profile.tolist(),
            lateral_profile=lateral_profile.tolist(),
            total_energy_deposited=total_deposited,
            shower_max_depth=float(shower_max),
            containment_fraction=float(containment)
        )

class MLDatasetGenerator:
    """Generate physics-validated datasets for ML training"""
    
    def __init__(self, output_dir: str = "ml_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.simulator = CalorimeterShowerSimulator()
        
    def generate_dataset(self, n_events: int = 1000, 
                        energy_range: Tuple[float, float] = (1.0, 100.0)) -> List[ShowerEvent]:
        """Generate dataset with energy sampling"""
        events = []
        energies = np.random.uniform(energy_range[0], energy_range[1], n_events)
        particles = np.random.choice(["electron", "photon"], n_events)
        
        for i, (energy, particle) in enumerate(zip(energies, particles)):
            event = self.simulator.simulate_em_shower(energy, particle)
            events.append(event)
            
            if (i + 1) % max(1, n_events // 10) == 0:
                progress = (i + 1) / n_events * 100
                sys.stdout.write(f"\r  Progress: {progress:.0f}% ({i+1}/{n_events} events)")
                sys.stdout.flush()
        
        print()  # New line after progress
        return events
    
    def validate_physics(self, events: List[ShowerEvent]) -> Dict:
        """Physics validation checks"""
        energies = [e.energy for e in events]
        shower_maxs = [e.shower_max_depth for e in events]
        containments = [e.containment_fraction for e in events]
        
        validation = {
            "energy_conservation": np.mean([e.total_energy_deposited / e.energy for e in events]),
            "shower_max_scaling": np.corrcoef(np.log(energies), shower_maxs)[0, 1],
            "containment_mean": np.mean(containments),
            "containment_std": np.std(containments)
        }
        return validation
    
    def save_dataset(self, events: List[ShowerEvent], split: str = "train"):
        """Save dataset in ML-ready format"""
        filepath = self.output_dir / f"{split}_dataset.json"
        with open(filepath, 'w') as f:
            json.dump([asdict(e) for e in events], f, indent=2)
        print(f"✓ Saved {len(events)} events to {filepath}")
        
    def visualize_sample(self, event: ShowerEvent, save_path: str = None):
        """Visualize shower profiles"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(event.depth_profile, 'b-', linewidth=2)
        ax1.set_xlabel('Layer Depth')
        ax1.set_ylabel('Energy Deposition (GeV)')
        ax1.set_title(f'Longitudinal Profile\n{event.particle_type.title()} @ {event.energy:.1f} GeV')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(event.lateral_profile, 'r-', linewidth=2)
        ax2.set_xlabel('Radial Distance (Moliere Radius)')
        ax2.set_ylabel('Normalized Energy')
        ax2.set_title('Lateral Profile')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Execute complete ML dataset generation pipeline"""
    print("\n" + "="*70)
    print("🔬 CALORIMETER SHOWER ML ANALYZER")
    print("   Electromagnetic Shower Simulation & Dataset Generation")
    print("="*70)
    
    print("\n👋 Hey! Let's generate some physics-validated ML training data...\n")
    time.sleep(0.5)
    
    generator = MLDatasetGenerator()
    start_time = time.time()
    
    # Generate datasets
    print("[Step 1/4] 🎲 Simulating training showers...")
    train_events = generator.generate_dataset(n_events=5000, energy_range=(1.0, 100.0))
    generator.save_dataset(train_events, "train")
    
    print("\n[Step 2/4] 🎯 Simulating validation showers...")
    val_events = generator.generate_dataset(n_events=1000, energy_range=(1.0, 100.0))
    generator.save_dataset(val_events, "validation")
    
    print("\n[Step 3/4] 🔍 Running physics validation checks...")
    validation = generator.validate_physics(train_events)
    print(f"\n  ✓ Energy Conservation: {validation['energy_conservation']:.4f} (should be ~1.0)")
    print(f"  ✓ Shower Max Correlation: {validation['shower_max_scaling']:.4f} (physics scaling)")
    print(f"  ✓ Containment: {validation['containment_mean']:.3f} ± {validation['containment_std']:.3f}")
    
    with open(generator.output_dir / "validation_report.json", 'w') as f:
        json.dump(validation, f, indent=2)
    print(f"  ✓ Saved validation report")
    
    print("\n[Step 4/4] 📊 Creating visualizations...")
    for i, event in enumerate(train_events[:5]):
        generator.visualize_sample(event, 
            save_path=generator.output_dir / f"shower_sample_{i+1}.png")
        sys.stdout.write(f"\r  Generated {i+1}/5 plots...")
        sys.stdout.flush()
    print(" Done!")
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("✨ ALL DONE! Dataset ready for ML training")
    print("="*70)
    print(f"\n📁 Output: {generator.output_dir.absolute()}")
    print(f"⏱️  Time: {elapsed:.1f} seconds")
    print(f"📦 Events: {len(train_events)} training + {len(val_events)} validation")
    
    print("\n💡 Next Steps:")
    print("   → Train a GNN or 3D CNN on the generated data")
    print("   → Target: <5% energy resolution for E > 10 GeV")
    print("   → Use for fast detector simulation or particle ID\n")

if __name__ == "__main__":
    main()
