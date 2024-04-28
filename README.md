# A Formal Perspective on Byte-Pair Encoding

> **Abstract:** Byte-Pair Encoding (BPE) is a popular algorithm used for tokenizing data in NLP, despite being devised initially as a compression method.BPE appears to be a greedy algorithm at face value, but the underlying optimization problem that BPE seeks to solve has not yet been laid down. We formalize BPE as a combinatorial optimization problem. Via submodular functions, we prove that the iterative greedy version is a 1/sigma*(1-e(-sigma))-approximation of an optimal merge sequence, where sigma is the total backward curvature with respect to the optimal merge sequence. Empirically the lower bound of the approximation is approx0.37.We provide a faster implementation of BPE which improves the runtime complexity from O(NM) to O(N log M), where N is the sequence length and M is the merge count. Finally, we optimize the brute-force algorithm for optimal BPE using memoization.

- read the [paper here](https://aclanthology.org/2023.findings-acl.38/),
- watch a [3 min video](https://www.youtube.com/watch?v=aB7oaS0rlvI),
- watch a [13-minute video](https://www.youtube.com/watch?v=yeEZpf4BlDA) of the paper and two others,
- or take a look [at the poster](meta/poster.pdf).

Cite as:
```
@inproceedings{zouhar2023formal, 
    title={A Formal Perspective on Byte-Pair Encoding},
    author={Zouhar, Vilém and Meister, Clara and Gastaldi, Juan Luis and Sachan, Du, Li and Vieira, Tim and Mrinmaya and Cotterell, Ryan},
    booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Findings)},
    year={2023},
    url={https://aclanthology.org/2023.findings-acl.38/},
}
```


[![Paper video presentation](https://img.youtube.com/vi/yeEZpf4BlDA/0.jpg)](https://www.youtube.com/watch?v=yeEZpf4BlDA)

[![Paper video presentation](https://img.youtube.com/vi/aB7oaS0rlvI/0.jpg)](https://www.youtube.com/watch?v=aB7oaS0rlvI)
