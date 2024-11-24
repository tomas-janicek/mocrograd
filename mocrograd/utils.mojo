from collections import Set


fn all_in[T: KeyElement](items: List[T], *, in_set: Set[T]) -> Bool:
    for item in items:
        if item[] not in in_set:
            return False
    return True
