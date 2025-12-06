"""
Tree Search Node for Tree-GRPO style optimization
Enhanced tree node class supporting batch sampling and tree-structured supervision signals.
"""
import math
import random
from typing import List, Optional


class TreeSearchNode:
    """
    Enhanced tree node for batch tree search with tree-structured supervision.
    Based on Tree-GRPO's TreeNode but adapted for the external codebase.
    """
    def __init__(
        self,
        node_id: str,
        state=None,
        question: str = None,
        parent=None,
        depth: int = 0,
        is_root: bool = False,
        reward_mode: str = 'base',
        margin: float = 0.1,
    ):
        self.node_id = node_id
        self.state = state.copy() if state else {}
        self.question = question
        self.parent = parent
        self.depth = depth
        self.is_root = is_root
        
        # Tree structure
        self._children = []
        
        # MCTS properties (for backward compatibility)
        self.visits = 0
        self.value = 0.0
        self.is_terminal = False
        self.reward = 0
        self.em = 0  # Exact match
        
        # Tree-GRPO properties
        self.original_score = 0.0
        self.final_score = 0.0
        self.subtree_leaf_score = 0.0
        self.reward_mode = reward_mode
        self.margin = margin
        self.is_leaf = False
        
        # Additional properties for compatibility
        self.solution = None  # For game24 compatibility
        
    @property
    def children(self):
        return self._children
    
    def add_child(self, child: 'TreeSearchNode'):
        """Add a child node."""
        if child is self:
            raise ValueError("A node cannot be its own child")
        self._children.append(child)
    
    def ucb_score(self, exploration_weight: float = 1.414) -> float:
        """Calculate UCB score for node selection.
        
        参考原始 LATS 的实现：当 visits == 0 时，返回 value 而不是 inf。
        这样可以确保未访问的节点也能根据其初始 value 进行比较。
        """
        if self.visits == 0:
            # 与原始 LATS 保持一致：返回 value 而不是 inf
            return self.value if self.value > 0 else 0.0
        
        parent_visits = self.parent.visits if self.parent else 1
        if parent_visits == 0:
            parent_visits = 1
        
        exploitation = self.value / self.visits if self.visits > 0 else 0
        exploration = exploration_weight * (2 * math.log(parent_visits) / self.visits) ** 0.5
        return exploitation + exploration
    
    def is_fully_expanded(self) -> bool:
        """
        Check if the node is fully expanded (has children and all children have been visited).
        Used for MCTS to determine if a node needs expansion.
        """
        return len(self._children) > 0 and all(child.visits > 0 for child in self._children)
    
    def get_subtree_nodes(self) -> List['TreeSearchNode']:
        """Get all descendant nodes by traversing the tree."""
        nodes = []
        nodes_to_visit = list(self._children)
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            nodes.append(current_node)
            nodes_to_visit.extend(current_node._children)
        return nodes
    
    def get_subtree_leaves(self) -> List['TreeSearchNode']:
        """Get all leaf nodes in the subtree."""
        leaves = []
        nodes_to_visit = list(self._children)
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            if current_node.is_leaf or len(current_node._children) == 0:
                leaves.append(current_node)
            nodes_to_visit.extend(current_node._children)
        return leaves
    
    def get_subtree_leaves_num(self) -> int:
        """Get the number of leaf nodes in the subtree."""
        return len(self.get_subtree_leaves())
    
    def get_expand_node(self, n: int = 1, mode: str = 'random') -> List['TreeSearchNode']:
        """
        Sample n nodes from the subtree for expansion.
        
        Args:
            n: Number of nodes to sample
            mode: Sampling mode ('random', 'best', 'uct')
        """
        candidate_set = []
        
        # Strategy 1: Include nodes that haven't been expanded yet (no children)
        # Include self if it's not a leaf and has no children yet
        if not self.is_terminal and self.depth < 7 and len(self._children) == 0:
            candidate_set.append(self)
        
        # Include all non-terminal nodes with no children from subtree
        for node in self.get_subtree_nodes():
            if not node.is_terminal and node.depth < 7 and len(node._children) == 0:
                candidate_set.append(node)
        
        # Strategy 2: If no unexpanded nodes found, select leaf nodes that are not terminal
        # This allows deeper exploration even after initial expansion
        if len(candidate_set) == 0:
            # Get all leaf nodes that are not terminal (can be expanded further)
            all_leaves = self.get_subtree_leaves()
            for leaf in all_leaves:
                # Include non-terminal leaves that haven't reached depth limit
                # Unmark as leaf to allow expansion
                if not leaf.is_terminal and leaf.depth < 7:
                    leaf.is_leaf = False  # Allow re-expansion
                    candidate_set.append(leaf)
        
        # Strategy 3: If still no candidates, try any non-terminal nodes that are marked as leaf
        # This handles cases where nodes were incorrectly marked as leaf
        if len(candidate_set) == 0:
            all_nodes = self.get_subtree_nodes() + [self]
            for node in all_nodes:
                if not node.is_terminal and node.depth < 7:
                    node.is_leaf = False  # Unmark to allow expansion
                    candidate_set.append(node)
        
        if len(candidate_set) == 0:
            return []
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for node in candidate_set:
            if node.node_id not in seen:
                seen.add(node.node_id)
                unique_candidates.append(node)
        candidate_set = unique_candidates
        
        if mode == 'random':
            result = random.choices(candidate_set, k=min(n, len(candidate_set)))
        elif mode == 'best':
            # Select nodes with highest value
            candidate_set.sort(key=lambda x: x.value, reverse=True)
            result = candidate_set[:min(n, len(candidate_set))]
        elif mode == 'uct':
            # Select nodes with highest UCT
            # 现在 ucb_score() 已经正确处理了 visits == 0 的情况，直接使用即可
            candidate_set.sort(key=lambda x: x.ucb_score(), reverse=True)
            result = candidate_set[:min(n, len(candidate_set))]
        else:
            result = random.choices(candidate_set, k=min(n, len(candidate_set)))
        
        return result
    
    def sample_leaf(self, n: int = 1) -> List['TreeSearchNode']:
        """
        Sample n leaves from the subtree, then prune the tree.
        
        Args:
            n: Number of leaves to sample
            
        Returns:
            List of sampled leaf nodes
        """
        candidate_leaves = self.get_subtree_leaves()
        
        if len(candidate_leaves) < n:
            # If not enough leaves, return all available
            return candidate_leaves
        
        random.shuffle(candidate_leaves)
        selected_leaves = candidate_leaves[:n]
        selected_ids = {leaf.node_id for leaf in selected_leaves}
        
        # Prune the tree to keep only paths to selected leaves
        self._prune_subtree(selected_ids)
        
        return selected_leaves
    
    def _prune_subtree(self, candidate_ids: set) -> bool:
        """
        Prune the subtree to keep only paths to nodes in candidate_ids.
        
        Args:
            candidate_ids: Set of node IDs to keep
            
        Returns:
            Whether this node should be kept
        """
        surviving_children = []
        
        for child in self._children:
            if child._prune_subtree(candidate_ids):
                surviving_children.append(child)
        
        self._children = surviving_children
        
        # Keep this node if it's in candidate_ids or still has children
        should_keep = self.node_id in candidate_ids or len(self._children) > 0
        
        return should_keep
    
    def set_leaf_original_score(self, score: float):
        """Set the original score of a leaf node."""
        self.original_score = score
    
    @staticmethod
    def dfs_subtree_leaf_score(node: 'TreeSearchNode') -> float:
        """
        Do DFS and compute the subtree leaf original score.
        
        Args:
            node: Root node of the subtree
            
        Returns:
            Sum of original scores of all leaves in the subtree
        """
        subtree_leaf_score = node.original_score if node.is_leaf else 0.0
        for child in node._children:
            subtree_leaf_score += TreeSearchNode.dfs_subtree_leaf_score(child)
        node.subtree_leaf_score = subtree_leaf_score
        return subtree_leaf_score
    
    def calculate_final_score_from_root(self):
        """
        Calculate the Diff-based final score from root.
        This implements Tree-GRPO's tree-structured supervision signal.
        """
        # First do DFS and compute the subtree leaf original score
        TreeSearchNode.dfs_subtree_leaf_score(self)
        
        # Then compute the final score for each node
        total_leaf_num = self.get_subtree_leaves_num()
        if total_leaf_num == 0:
            return
        
        global_score_mean = self.subtree_leaf_score / total_leaf_num
        
        for node in self.get_subtree_nodes():
            if node.is_leaf:
                curr_leaf_num = 1
            else:
                curr_leaf_num = node.get_subtree_leaves_num()
            
            if curr_leaf_num == 0:
                continue
            
            curr_score_mean = node.subtree_leaf_score / curr_leaf_num
            
            # Tree-GRPO uses base mode (no diff calculation)
            if self.reward_mode == 'tree_diff':
                # TreeRL global score
                global_score = curr_score_mean - global_score_mean
                global_score = 0.0  # Disabled for Tree-GRPO
                
                # TreeRL local score
                if node.parent:
                    parent_leaf_num = node.parent.get_subtree_leaves_num()
                    if parent_leaf_num > 0:
                        parent_score_mean = node.parent.subtree_leaf_score / parent_leaf_num
                        local_score = curr_score_mean - parent_score_mean
                        diff_score = global_score + local_score
                        diff_score = max(diff_score - self.margin, 0.0)
                        final_score = diff_score + curr_score_mean
                    else:
                        final_score = curr_score_mean
                else:
                    final_score = curr_score_mean
                
                node.final_score = final_score / math.sqrt(curr_leaf_num)
            else:
                # Base mode: use original score directly
                node.final_score = curr_score_mean
    
    def collect_trajectory(self) -> List:
        """
        Collect the trajectory from root to this node.
        
        Returns:
            List of states from root to this node
        """
        trajectory = []
        node = self
        while node:
            if node.state:
                trajectory.insert(0, node.state)
            node = node.parent
        return trajectory
    
    def __str__(self):
        if hasattr(self, 'solution') and self.solution:
            return f"TreeSearchNode(id={self.node_id[:8]}, depth={self.depth}, value={self.value:.2f}, solution={self.solution[:30]})"
        return f"TreeSearchNode(id={self.node_id[:8]}, depth={self.depth}, value={self.value:.2f}, reward={self.reward})"

