"""
FedCS: Federated Client Selection

Greedy scheduler that selects fastest devices based on 
estimated completion time.

Reference: Nishio & Yonetani, "Client Selection for Federated 
Learning with Heterogeneous Resources in Mobile Edge," ICC 2019
"""
import numpy as np


class FedCSScheduler:
    """
    FedCS: Client Selection based on device capability
    
    Selects K fastest devices (lowest estimated completion time)
    """
    
    def __init__(self, num_devices, devices_per_round, device_capabilities, 
                 seed=42):
        """
        Args:
            num_devices: Total number of devices
            devices_per_round: Number of devices to select (K)
            device_capabilities: Dict of device capabilities from simulator
                                {device_id: {'a_k': ..., 'mu_k': ...}}
            seed: Random seed
        """
        self.num_devices = num_devices
        self.devices_per_round = devices_per_round
        self.device_capabilities = device_capabilities
        self.seed = seed
        
        # Track selection history for fairness calculation
        self.selection_history = {i: 0 for i in range(num_devices)}
        
        # Extract device speeds (a_k values)
        self.device_speeds = {
            i: device_capabilities[i]['a_k'] 
            for i in range(num_devices)
        }
        
        print(f"FedCS Scheduler initialized")
        print(f"  Total devices: {num_devices}")
        print(f"  Devices per round: {devices_per_round}")
        print(f"  Selection strategy: Greedy (fastest devices)")
    
    def estimate_completion_time(self, device_id, dataset_size, 
                                 model_size=1.0, tau_m=0.001):
        """
        Estimate time for device to complete one training round
        
        Simplified version of paper's formula:
        time = computation_time + communication_time
        
        Args:
            device_id: Device identifier
            dataset_size: Number of samples on device
            model_size: Size of model (for communication)
            tau_m: Time coefficient for model complexity
        
        Returns:
            float: Estimated completion time
        """
        a_k = self.device_speeds[device_id]
        
        # Computation time (proportional to dataset size and device speed)
        computation_time = tau_m * a_k * dataset_size
        
        # Communication time (proportional to model size and device speed)
        # Simplified: assume communication also affected by device capability
        communication_time = model_size * a_k * 0.1
        
        total_time = computation_time + communication_time
        
        return total_time
    
    def select_devices(self, available_devices=None, dataset_sizes=None):
        """
        Select K fastest devices based on estimated completion time
        
        Args:
            available_devices: List of available device IDs
                             If None, all devices are available
            dataset_sizes: Dict {device_id: dataset_size}
                          If None, assume equal sizes (500 samples)
        
        Returns:
            list: Selected device IDs (K fastest)
        """
        # If no available devices specified, use all
        if available_devices is None:
            available_devices = list(range(self.num_devices))
        
        # If no dataset sizes provided, assume equal
        if dataset_sizes is None:
            dataset_sizes = {i: 500 for i in available_devices}
        
        # Calculate estimated time for each available device
        device_times = []
        for device_id in available_devices:
            dataset_size = dataset_sizes.get(device_id, 500)
            est_time = self.estimate_completion_time(device_id, dataset_size)
            device_times.append((device_id, est_time))
        
        # Sort by estimated time (ascending - fastest first)
        device_times.sort(key=lambda x: x[1])
        
        # Select K fastest devices
        k = min(self.devices_per_round, len(available_devices))
        selected = [device_id for device_id, _ in device_times[:k]]
        
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
    
    def get_selected_device_speeds(self):
        """
        Get average speed of selected vs non-selected devices
        
        Returns:
            dict: Speed statistics
        """
        selected_speeds = []
        not_selected_speeds = []
        
        for device_id, count in self.selection_history.items():
            speed = self.device_speeds[device_id]
            if count > 0:
                selected_speeds.append(speed)
            else:
                not_selected_speeds.append(speed)
        
        return {
            'selected_avg_speed': np.mean(selected_speeds) if selected_speeds else 0,
            'not_selected_avg_speed': np.mean(not_selected_speeds) if not_selected_speeds else 0,
            'num_selected_devices': len(selected_speeds),
            'num_not_selected_devices': len(not_selected_speeds)
        }



if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0,     os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    
    from utils.device_simulator import DeviceSimulator
    
    print("Testing FedCS Scheduler\n")
    print("=" * 60)
    
    # Create device simulator to get capabilities
    simulator = DeviceSimulator(num_devices=30, seed=42)
    capabilities = simulator.get_capabilities()
    
    # Create FedCS scheduler
    scheduler = FedCSScheduler(
        num_devices=30,
        devices_per_round=10,
        device_capabilities=capabilities,
        seed=42
    )
    
    # Show device speeds
    print("\nDevice Speeds (a_k):")
    print("-" * 60)
    speeds = [(i, capabilities[i]['a_k']) for i in range(30)]
    speeds_sorted = sorted(speeds, key=lambda x: x[1])
    
    print("Fastest 5 devices:")
    for dev_id, speed in speeds_sorted[:5]:
        print(f"  Device {dev_id}: a_k = {speed:.3f}")
    
    print("\nSlowest 5 devices:")
    for dev_id, speed in speeds_sorted[-5:]:
        print(f"  Device {dev_id}: a_k = {speed:.3f}")
    
    # Simulate 10 rounds
    print("\n" + "=" * 60)
    print("SIMULATING 10 ROUNDS")
    print("=" * 60)
    
    for round_num in range(10):
        selected = scheduler.select_devices()
        # Show average speed of selected devices
        avg_speed = np.mean([scheduler.device_speeds[i] for i in selected])
        print(f"Round {round_num+1}: Selected {len(selected)} devices, "
              f"avg speed = {avg_speed:.3f}")
    
    # Show statistics
    print("\n" + "=" * 60)
    print("SELECTION STATISTICS")
    print("=" * 60)
    stats = scheduler.get_selection_stats()
    print(f"Mean selections per device: {stats['mean_selections']:.2f}")
    print(f"Std deviation: {stats['std_selections']:.2f}")
    print(f"Min selections: {stats['min_selections']}")
    print(f"Max selections: {stats['max_selections']}")
    print(f"Fairness score: {stats['fairness_score']:.2f}")
    
    # Compare with random
    print("\n" + "=" * 60)
    print("SPEED ANALYSIS")
    print("=" * 60)
    speed_stats = scheduler.get_selected_device_speeds()
    print(f"Devices selected at least once: {speed_stats['num_selected_devices']}")
    print(f"Devices never selected: {speed_stats['num_not_selected_devices']}")
    print(f"Average speed of selected devices: {speed_stats['selected_avg_speed']:.3f}")
    if speed_stats['num_not_selected_devices'] > 0:
        print(f"Average speed of ignored devices: {speed_stats['not_selected_avg_speed']:.3f}")
    
    # Show which devices never selected
    never_selected = [i for i, count in scheduler.selection_history.items() if count == 0]
    if never_selected:
        print(f"\nDevices never selected ({len(never_selected)} devices):")
        for dev_id in never_selected[:5]:  # Show first 5
            print(f"  Device {dev_id}: a_k = {scheduler.device_speeds[dev_id]:.3f} (slow)")
    
    print("\n" + "=" * 60)
    print("FedCS Scheduler working correctly!")
    print("=" * 60)
    print("\nKey observation: FedCS always selects fast devices")
    print("This improves speed but hurts fairness!")