"""
Random Scheduler

Baseline scheduler: Randomly selects devices for each round
"""
import numpy as np

class RandomScheduler:
    """
    Random device selection scheduler
    
    Selects K devices uniformly at random from available devices
    """
    
    def __init__(self, num_devices, devices_per_round, seed=42):
        """
        Args:
            num_devices: Total number of devices
            devices_per_round: Number of devices to select (K)
            seed: Random seed for reproducibility
        """
        self.num_devices = num_devices
        self.devices_per_round = devices_per_round
        self.seed = seed
        
        # Initialize random generator
        self.rng = np.random.RandomState(seed)
        
        # Track selection history for fairness calculation
        self.selection_history = {i: 0 for i in range(num_devices)}
        
        print(f"Random Scheduler initialized")
        print(f"  Total devices: {num_devices}")
        print(f"  Devices per round: {devices_per_round}")
    
    def select_devices(self, available_devices=None):
        """
        Select devices randomly
        
        Args:
            available_devices: List of available device IDs
                             If None, all devices are available
        
        Returns:
            list: Selected device IDs
        """
        # If no available devices specified, use all
        if available_devices is None:
            available_devices = list(range(self.num_devices))
        
        # Make sure we don't try to select more than available
        k = min(self.devices_per_round, len(available_devices))
        
        # Random selection without replacement
        selected = self.rng.choice(
            available_devices,
            size=k,
            replace=False
        ).tolist()
        
        # Update selection history
        for device_id in selected:
            self.selection_history[device_id] += 1
        
        return selected
    
    def get_fairness_score(self):
        """
        Calculate fairness score (standard deviation of selection frequency)
        Lower is better (more fair)
        
        Returns:
            float: Standard deviation of selection counts
        """
        selection_counts = list(self.selection_history.values())
        return np.std(selection_counts)
    
    def get_selection_stats(self):
        """
        Get selection statistics
        
        Returns:
            dict: Statistics about device selection
        """
        counts = list(self.selection_history.values())
        return {
            'mean_selections': np.mean(counts),
            'std_selections': np.std(counts),
            'min_selections': np.min(counts),
            'max_selections': np.max(counts),
            'fairness_score': self.get_fairness_score()
        }


# Test
if __name__ == "__main__":
    print("Testing Random Scheduler\n")
    print("=" * 60)
    
    # Create scheduler
    scheduler = RandomScheduler(
        num_devices=30,
        devices_per_round=10,
        seed=42
    )
    
    # Simulate 10 rounds
    print("\nSimulating 10 rounds of device selection:")
    print("-" * 60)
    
    for round_num in range(10):
        selected = scheduler.select_devices()
        print(f"Round {round_num+1}: {selected}")
    
    # Show fairness statistics
    print("\n" + "=" * 60)
    print("SELECTION STATISTICS")
    print("=" * 60)
    stats = scheduler.get_selection_stats()
    print(f"Mean selections per device: {stats['mean_selections']:.2f}")
    print(f"Std deviation: {stats['std_selections']:.2f}")
    print(f"Min selections: {stats['min_selections']}")
    print(f"Max selections: {stats['max_selections']}")
    print(f"Fairness score: {stats['fairness_score']:.2f}")
    
    # Show which devices selected most/least
    selection_counts = scheduler.selection_history
    sorted_devices = sorted(selection_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMost selected devices:")
    for dev_id, count in sorted_devices[:5]:
        print(f"  Device {dev_id}: {count} times")
    
    print("\nLeast selected devices:")
    for dev_id, count in sorted_devices[-5:]:
        print(f"  Device {dev_id}: {count} times")
    
    print("\n" + "=" * 60)
    print("Random Scheduler working correctly!")
    print("=" * 60)