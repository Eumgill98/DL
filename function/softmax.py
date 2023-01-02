import numpy as np

def softmax(vec): #input : 벡터
    denumerator = np.exp(vec - np.max(vec, axis=-1, keepdims = True)) #오버 플로우 방지를 위해서 최댓값을 빼준다
    numerator = np.sum(denumerator, axis=-1, keepdims=True)
    val = denumerator / numerator
    return val