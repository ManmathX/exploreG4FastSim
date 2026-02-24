#!/usr/bin/env python3
"""Quick demo to explore the generated ML dataset"""

import json
import numpy as np

def explore_dataset():
    print("\n🔍 Dataset Explorer\n" + "="*50)
    
    with open("ml_dataset/train_dataset.json", 'r') as f:
        data = json.load(f)
    
    print(f"\n📊 Dataset Stats:")
    print(f"   Total events: {len(data)}")
    
    energies = [e['energy'] for e in data]
    print(f"   Energy range: {min(energies):.1f} - {max(energies):.1f} GeV")
    
    particles = [e['particle_type'] for e in data]
    print(f"   Electrons: {particles.count('electron')}")
    print(f"   Photons: {particles.count('photon')}")
    
    print(f"\n🎯 Sample Event:")
    sample = data[0]
    print(f"   Energy: {sample['energy']:.2f} GeV")
    print(f"   Particle: {sample['particle_type']}")
    print(f"   Shower max depth: {sample['shower_max_depth']:.2f}")
    print(f"   Energy deposited: {sample['total_energy_deposited']:.2f} GeV")
    print(f"   Containment: {sample['containment_fraction']:.3f}")
    print(f"   Profile points: {len(sample['depth_profile'])} depth + {len(sample['lateral_profile'])} lateral")
    
    print("\n✅ Ready for ML training!\n")

if __name__ == "__main__":
    explore_dataset()
