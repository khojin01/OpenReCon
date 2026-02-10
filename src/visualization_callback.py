import numpy as np
from collections import defaultdict
from typing import Dict, List, Set

class RefinementTracker:
    """Tracks refinement process for visualization"""
    
    def __init__(self):
        self.history = []
        self.current_iteration = 0
        self.reset()
    
    def reset(self):
        """Reset tracking for new run"""
        self.history = []
        self.current_iteration = 0
        self.iteration_data = {
            'iteration': 0,
            'stage': 'initial',
            'clusters': None,
            'purged_nodes': set(),
            'imported_nodes': {},  # node_id -> cluster_id
            'refined_nodes': {},   # node_id -> (old_cluster, new_cluster)
            'cluster_sizes': {},
            'metrics': {},
            'embeddings': None  # Store embeddings for this iteration
        }
    
    def start_iteration(self, iteration: int):
        """Start tracking a new iteration"""
        if self.iteration_data['clusters'] is not None:
            self.history.append(self.iteration_data.copy())
        
        self.current_iteration = iteration
        self.iteration_data = {
            'iteration': iteration,
            'stage': 'start',
            'clusters': None,
            'purged_nodes': set(),
            'imported_nodes': {},
            'refined_nodes': {},
            'cluster_sizes': {},
            'metrics': {},
            'embeddings': None
        }
    
    def record_initial_state(self, clusters: np.ndarray):
        """Record initial cluster state"""
        self.iteration_data['clusters'] = clusters.copy()
        self.iteration_data['cluster_sizes'] = self._compute_cluster_sizes(clusters)
    
    def record_embeddings(self, embeddings: np.ndarray):
        """Record node embeddings for this iteration"""
        self.iteration_data['embeddings'] = embeddings.copy() if embeddings is not None else None
    
    def record_purge(self, purged_nodes: Set[int], clusters_before: np.ndarray, clusters_after: np.ndarray):
        """Record purged nodes"""
        self.iteration_data['purged_nodes'] = purged_nodes.copy()
        self.iteration_data['stage'] = 'purge'
        self.iteration_data['clusters_before_purge'] = clusters_before.copy()
        self.iteration_data['clusters_after_purge'] = clusters_after.copy()
    
    def record_import(self, imported_nodes: Dict[int, int], clusters_after: np.ndarray):
        """Record imported nodes"""
        self.iteration_data['imported_nodes'] = imported_nodes.copy()
        self.iteration_data['stage'] = 'import'
        self.iteration_data['clusters_after_import'] = clusters_after.copy()
    
    def record_refinement(self, refined_nodes: Dict[int, tuple], clusters_before: np.ndarray, clusters_after: np.ndarray):
        """Record refined nodes (label propagation changes)"""
        self.iteration_data['refined_nodes'] = refined_nodes.copy()
        self.iteration_data['stage'] = 'refinement'
        self.iteration_data['clusters_before_refinement'] = clusters_before.copy()
        self.iteration_data['clusters_after_refinement'] = clusters_after.copy()
    
    def record_metrics(self, metrics: Dict):
        """Record metrics for this iteration"""
        self.iteration_data['metrics'] = metrics.copy()
    
    def finalize_iteration(self):
        """Finalize current iteration"""
        self.history.append(self.iteration_data.copy())
    
    def _compute_cluster_sizes(self, clusters: np.ndarray) -> Dict[int, int]:
        """Compute size of each cluster"""
        unique, counts = np.unique(clusters[clusters != -1], return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def get_history(self) -> List[Dict]:
        """Get full history"""
        return self.history
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.history:
            return {}
        
        total_purged = sum(len(it['purged_nodes']) for it in self.history)
        total_imported = sum(len(it['imported_nodes']) for it in self.history)
        total_refined = sum(len(it['refined_nodes']) for it in self.history)
        
        return {
            'total_iterations': len(self.history),
            'total_purged': total_purged,
            'total_imported': total_imported,
            'total_refined': total_refined,
            'iterations': self.history
        }

# Global tracker instance
_global_tracker = None

def get_tracker() -> RefinementTracker:
    """Get global tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = RefinementTracker()
    return _global_tracker

def reset_tracker():
    """Reset global tracker"""
    global _global_tracker
    _global_tracker = RefinementTracker()
    return _global_tracker
