"""References used throughout the package."""
from .due import BibTeX

GODEC = BibTeX(
    """
    @InProceedings{ICML2011Zhou_41,
        author =    {Tianyi Zhou and Dacheng Tao},
        title =     {GoDec: Randomized Low-rank & Sparse Matrix Decomposition in Noisy Case },
        booktitle = {Proceedings of the 28th International Conference on Machine Learning
                     (ICML-11)},
        series =    {ICML '11},
        year =      {2011},
        editor =    {Lise Getoor and Tobias Scheffer},
        location =  {Bellevue, Washington, USA},
        isbn =      {978-1-4503-0619-5},
        month =     {June},
        publisher = {ACM},
        address =   {New York, NY, USA},
        pages=      {33--40},
    }
    """
)

BILATERAL_SKETCH = BibTeX(
    """
    @article{zhou2013greedy,
        title={Greedy bilateral sketch, completion and smoothing for large-scale matrix completion,
               robust PCA and low-rank approximation},
        author={Zhou, T and Tao, D},
        journal={AISTATS 2013},
        year={2013}
    }
    """
)
