import heapq
import re
from difflib import SequenceMatcher

# -----------------------
# Utility functions
# -----------------------

def sentence_tokenize(text):
    """Split text into sentences using basic punctuation rules."""
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

def similarity(a, b):
    """Compute a simple similarity score between two sentences (0–1)."""
    return SequenceMatcher(None, a, b).ratio()

# -----------------------
# A* Search Implementation
# -----------------------

class Node:
    def __init__(self, i, j, cost, path):
        self.i = i  # position in doc1
        self.j = j  # position in doc2
        self.cost = cost
        self.path = path  # alignment path (list of tuples)

    def __lt__(self, other):
        return self.cost < other.cost


def heuristic(i, j, len1, len2):
    """Simple heuristic: remaining unmatched sentences."""
    return abs((len1 - i) - (len2 - j))


def a_star_align(doc1, doc2):
    len1, len2 = len(doc1), len(doc2)
    start = Node(0, 0, 0, [])
    pq = [(0, start)]

    visited = set()

    while pq:
        _, node = heapq.heappop(pq)
        i, j = node.i, node.j

        if (i, j) in visited:
            continue
        visited.add((i, j))

        # If we reached the end of both documents
        if i == len1 and j == len2:
            return node.path

        # Explore next states
        if i < len1 and j < len2:
            sim = similarity(doc1[i], doc2[j])
            # Lower cost means higher similarity
            cost = node.cost + (1 - sim)
            new_path = node.path + [((i, doc1[i]), (j, doc2[j]), sim)]
            heapq.heappush(pq, (cost + heuristic(i + 1, j + 1, len1, len2),
                                 Node(i + 1, j + 1, cost, new_path)))

        # Skip a sentence from doc1
        if i < len1:
            heapq.heappush(pq, (node.cost + 1 + heuristic(i + 1, j, len1, len2),
                                 Node(i + 1, j, node.cost + 1, node.path)))

        # Skip a sentence from doc2
        if j < len2:
            heapq.heappush(pq, (node.cost + 1 + heuristic(i, j + 1, len1, len2),
                                 Node(i, j + 1, node.cost + 1, node.path)))

    return []


# -----------------------
# Main Function
# -----------------------

def detect_plagiarism(text1, text2, threshold=0.8):
    doc1 = sentence_tokenize(text1)
    doc2 = sentence_tokenize(text2)

    alignment = a_star_align(doc1, doc2)

    print("\n--- Potentially Plagiarized Sentences ---\n")
    for ((i, s1), (j, s2), sim) in alignment:
        if sim >= threshold:
            print(f"[Doc1: Sentence {i+1}] {s1}")
            print(f"[Doc2: Sentence {j+1}] {s2}")
            print(f"→ Similarity: {sim:.2f}")
            print("-" * 60)


# -----------------------
# Example Usage
# -----------------------

if __name__ == "__main__":
    text1 = """Artificial Intelligence is transforming the world.
    It has applications in every field.
    Machine learning is a core part of AI.
    Plagiarism detection helps maintain academic honesty."""

    text2 = """AI is changing the world significantly.
    Machine learning forms the foundation of artificial intelligence.
    Detecting plagiarism ensures honesty in academics.
    Robotics is another application of AI."""

    detect_plagiarism(text1, text2)