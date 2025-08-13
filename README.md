# An Implicit Adaptive Fourier Neural Operator for Long-term Predictions of Three-dimensional Turbulence

Code and data accompanying the manuscript titled ["An Implicit Adaptive Fourier Neural Operator for Long-term Predictions of Three-dimensional Turbulence"](https://arxiv.org/abs/2501.12740), authored by Yuchi Jiang, Zhijie Li, Yunpeng Wang, Huiyu Yang and Jianchun Wang.

## Abstract

Long-term prediction of three-dimensional (3D) turbulent flows is one of the most challenging problems for machine learning approaches. Although some existing machine learning approaches such as implicit U-net enhanced Fourier neural operator (IUFNO) have been proven to be capable of achieving stable long-term predictions for turbulent flows, their computational costs are usually high. In this paper, we use the adaptive Fourier neural operator (AFNO) as the backbone to construct a model that can predict 3D turbulence. Furthermore, we employ the implicit iteration to our constructed AFNO and propose the implicit adaptive Fourier neural operator (IAFNO). IAFNO is systematically tested in three types of 3D turbulence, including forced homogeneous isotropic turbulence (HIT), temporally evolving turbulent mixing layer and turbulent channel flow. The numerical results demonstrate that IAFNO is more accurate than IUFNO and the traditional large-eddy simulation using dynamic Smagorinsky model (DSM), while exhibiting greater stability compared to IUFNO. Meanwhile, the AFNO model exhibits instability in numerical simulations. Moreover, the training efficiency of IAFNO is 4 times higher than that of IUFNO, and the number of parameters and GPU memory occupation of IAFNO are only 1/80 and 1/3 of IUFNO, respectively in HIT. In other tests, the improvements are slightly lower but still considerable. These improvements mainly come from patching and self-attention in 3D space. Besides, the well-trained IAFNO is significantly faster than the DSM.

## Dataset

The dataset can download at [fDNS_kaggle](https://www.kaggle.com/datasets/aifluid/coarsened-fdns-data-chl).



## Citation

arXiv version:
```
@article{jiang2025implicit,
  title={An implicit adaptive Fourier neural operator for long-term predictions of three-dimensional turbulence},
  author={Jiang, Yuchi and Li, Zhijie and Wang, Yunpeng and Yang, Huiyu and Wang, Jianchun},
  journal={arXiv preprint arXiv:2501.12740},
  year={2025}
}
```

This manuscript has been accepted by Acta Mechanica Sinica with an assigned DOI: 10.1007/s10409-025-25478-x. When the final version of the article provided by the journal becomes retrievable, please cite it using the information of the final version. Many thanks.
