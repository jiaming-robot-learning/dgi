## Implementation of DGI
In this repository we implemented the DGI (Veličković et al., 2023) framework and tested on the [Moleculenet](https://moleculenet.org/datasets-1) HIV dataset.

One drawback of the original proposed method is that it is only used in the unsupervised learning setting; however, many dataset do have lebels, e.g. the PPI dataset used in the original paper. Thus the information is not fully utilized in this setting. To overcome this, we empirically showed that the proposed infomax framework can improve the performance in supervised learning settings by simply combine the regression loss with the infomax loss.


### To reproduce reported result

Install required dependencies:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

Run the script:
```
python main.py
```

## References

- Veličković, P., Fedus, W., Hamilton, W. L., Liò, P., Bengio, Y., & Hjelm, R. D. (2023, January 23). Deep Graph Infomax. International Conference on Learning Representations. https://openreview.net/forum?id=rklz9iAcKQ
- Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., & Pande, V. (2018). MoleculeNet: A benchmark for molecular machine learning. Chemical Science, 9(2), 513–530. https://doi.org/10.1039/C7SC02664A