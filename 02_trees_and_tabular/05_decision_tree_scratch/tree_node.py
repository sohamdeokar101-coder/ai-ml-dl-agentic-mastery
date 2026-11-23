class Node:
    """
    A single node in the Decision Tree.
    
    If it's an internal decision node:
        - feature: index of the feature used for splitting (x_j)
        - threshold: numerical cutoff value (x_j <= threshold)
        - left: left child Node pointer (True condition)
        - right: right child Node pointer (False condition)
        - info_gain: information gain achieved by this split
        
    If it's a leaf node:
        - value: majority class label prediction
    """
    def __init__(
        self,
        feature: int = None,
        threshold: float = None,
        left=None,
        right=None,
        info_gain: float = 0.0,
        value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

    def is_leaf(self) -> bool:
        """Returns True if this node is a terminal leaf node."""
        return self.value is not None
