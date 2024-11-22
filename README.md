**This branch includes the Python implementation of directed transfer function (DTF). Major library includes: Numpy, Panda and Statsmodel**. 

DTF is a frequency-domain directed (effective) connectivity metric for brain signal analysis and functional network modeling. It is based on multi variable autoregressive model (MVAR) over frequency band (e.g., 10-30Hz). introduction can be found in:

**M. Kamiński and K.J. Blinowska, "A new method of the description of the information flow in the brain structures", Biol. Cybern. 65, 1991, pp 203-210.**
**A. Korzeniewska, M. Mańczak, M. Kamiński, K.J. Blinowska, S. Kasicki, "Determination of information flow direction among brain structures by a modified directed transfer function (dDTF) method", J. Neurosci. Methods 125 (1–2) (2003)
pp 195–207.**

For illustration purpose, there are 20 signal channels (i.e., SEEG contact points), the model order K is chosen 10, and frequency band is $$\beta$$ [13,30]Hz. Hence, the output is 20 by 20 DTF matrix over beta band.

All SEEG-related information is anonymous and deidentified due to IRB regulation of UT Dallas and UTSW.
