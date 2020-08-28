Supervised Autoencoder for Language Detection
----------------------------------------------

The Supervised Autoencoder (SAE) with Bayesian Optimization (BO) for the language detection task found effectively for discriminating between very close languages or dialects. This library contains the PyTorch implementation of SAE with one sample code for using it for the language detection task. The library can be used for other NLP classification tasks (e.g. Fake News Detection, Operant Motive Detection) easily. It supports both CPU and GPU versions with just turn on/off the GPU flag ("is_gpu = True or False"). 


Dependency
----------

Python 3.6

PyTorch (torch=1.0.1, torchtext=0.4.0, torchvision=0.4.0)

Utilities (Skopt, sklearn, numpy, Zipfile, Pandas, Pickel) 


Reference Paper
---------------

[1] Parida, S., ú Villatoro-Tello, E., Kumar, S., Motlicek, P., & Zhan, Q. (2020). Idiap Submission to Swiss-German Language Detection Shared Task. In Proceedings of the 5th Swiss Text Analytics Conference (SwissText) & 16th Conference on Natural Language Processing (KONVENS).

[2] Le, L., Patterson, A., & White, M. (2018). Supervised autoencoders: Improving generalization performance with unsupervised regularizers. In Advances in Neural Information Processing Systems (pp. 107-117).

[3] Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical bayesian optimization of machine learning algorithms. In Advances in neural information processing systems (pp. 2951-2959).


Citation
--------

If you found our research helpful or influential please consider citing

@inproceedings{parida2020idiap,
  title={Idiap Submission to Swiss-German Language Detection Shared Task},
  author={Parida, Shantipriya and {\'u} Villatoro-Tello, Esa and Kumar, Sajit and Motlicek, Petr and Zhan, Qingran},
  booktitle={Proceedings of the 5th Swiss Text Analytics Conference (SwissText) \& 16th Conference on Natural Language Processing (KONVENS)},
  year={2020}
}

Acknowledgement
---------------

The work was supported by an innovation project (under an InnoSuisse grant) oriented to improvethe automatic speech recognition and natural language understanding technologies for German. Title:  “SM2:  Extracting  Semantic  Meaning  from Spoken Material” funding application no. 29814.1IP-ICT and EU H2020 project “Real-time network,text, and speaker analytics for combating organizedcrime” (ROXANNE), grant agreement:  833635.The  second  author,  Esa ́u  Villatoro-Tello  is  sup-ported  partially  by  Idiap,  UAM-C  Mexico,  an dSNI-CONACyT Mexico during the elaboration ofthis work.