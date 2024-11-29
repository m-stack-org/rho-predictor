# ρ-predictor

A tool to predict the electronic density of molecules using a SA-GPR model

## Requirements

- [Q-stack](https://github.com/lcmd-epfl/Q-stack)
- [metatensor](https://github.com/metatensor/metatensor)
- [featomic](https://github.com/metatensor/featomic)

## Installation

You can install it by runing the comand:

```
pip install git+https://github.com/m-stack-org/rho-prediction.git
```
## Before usage

Please be sure you have the weights and averages of a pre-trained model.

TODO: how to add the data


## Usage
In a script
```
import rho_predictor
rho_predictor.predictor.predict_sagpr('path/to/mol.xyz', 'bfdb_HCNO')
```
or as a cli tool:
```
python -m  rho_predictor.predictor path/to/mol.xyz bfdb_HCNO
```

## Acknowledgements

The authors acknowledge the National Centre of Competence in Research (NCCR) "Materials' Revolution: Computational Design
and Discovery of Novel Materials (MARVEL)" of the Swiss National Science Foundation (SNSF, grant number 182892) and the
European Research Council (ERC, grant agreement no 817977).

## References

<ol>

### Theory

<li>
<a href='https://doi.org/10.1063/5.0055393' target="_blank" rel="noopener noreferrer">                                        
Briling, K. R.; Fabrizio, A.; Corminboeuf, C. Impact of Quantum-Chemical Metrics                                              
on the Machine Learning Prediction of Electron Density. <i> J. Chem. Phys.                                                    
</i> <b> 2021 </b>, <i>155</i>, 024107;                                                                                       
doi: 10.1063/5.0055393                                                                                                        
</a>.                                  
</li>

<li>
<a href='https://doi.org/10.1063/5.0033326' target="_blank" rel="noopener noreferrer">                                        
Fabrizio, A.; Briling, K. R.; Girardier, D. D.; Corminboeuf, C.  Learning                                                     
On-Top: Regressing the On-Top Pair Density for Real-Space Visualization of                                                    
Electron Correlation. <i> J. Chem. Phys. </i> <b> 2020 </b>,                                                                  
<i>153</i>, 204111;                                                                                                           
doi: 10.1063/5.0033326                                                                                                        
</a>.                                                                                                                         
</li>
                                                                                                                              
<li>
<a href='https://doi.org/10.1039/C9SC02696G' target="_blank" rel="noopener noreferrer">                                       
Fabrizio, A.; Grisafi, A.; Meyer, B.; Ceriotti, M.; Corminboeuf, C. Electron                                                  
Density Learning of Non-Covalent Systems. <i>Chem. Sci.</i>  <b>2019</b>,                                                     
<i>10</i>, 9424;                                                                                                              
doi: 10.1039/C9SC02696G                                                                                                       
</a>.                                                                                                                         
</li>
                                                                                                                              
                                                                                                                              
<li>
<a href='https://doi.org/10.1021/acscentsci.8b00551' target="_blank" rel="noopener noreferrer">                               
Grisafi, A.; Fabrizio, A.; Meyer, B.; Wilkins, D. M.; Corminboeuf, C.; Ceriotti,                                              
M. Transferable Machine-Learning Model of the Electron Density. <i>ACS Cent. Sci.</i> <b> 2019 </b>, <i>5</i>, 57;            
doi: 10.1021/acscentsci.8b00551                                                                                               
</a>.                                                                                                                         
</li>

### Representations

<li>                                                                                                                          
<a href="https://doi.org/10.1103/physrevlett.120.036002">                                                                     
Grisafi, A.; Wilkins, D. M.; Csányi, G.; Ceriotti, M. Symmetry-Adapted                                                        
Machine Learning for Tensorial Properties of Atomistic Systems.<i> Phys. Rev. Lett. </i> <b>2018</b>, <i>120</i>, 036002;     
doi: 10.1103/physrevlett.120.036002                                                                                           
</a>.                                                                                                                         
</li>                                                                                                                         
                                                                                                                              
<li>                                                                                                                          
<a href="https://doi.org/10.1103/physrevb.87.184115">                                                                         
Bartók, A. P.; Kondor, R.; Csányi, G. On Representing Chemical Environments.                                                  
<i>Phys. Rev. B</i> <b>2013</b>, <i>87</i>;                                                                                   
doi: 10.1103/physrevb.87.184115                                                                                               
</a>.                                                                                                                         
</li>                                      

### Examples of applications

<li>                                                                                                                          
<a href='https://doi.org/10.1021/acs.jpclett.1c01425' target="_blank" rel="noopener noreferrer">                              
Vela, S.; Fabrizio, A.; Briling, K. R.; Corminboeuf, C. Learning the Exciton                                                  
Properties of Azo-Dyes. <i>J. Phys. Chem. Lett.</i> <b>                                                                       
2021 </b>, <i>12</i>, 5957;                                                                                                   
doi: 10.1021/acs.jpclett.1c01425                                                                                              
</a>.                                                                                                                         
</li>                                                                                                                         
                                                                                                                              
<li>                                                                                                                          
<a href='https://doi.org/10.2533/chimia.2020.232' target="_blank" rel="noopener noreferrer">                                  
Fabrizio, A.; Briling, K.; Grisafi, A.; Corminboeuf, C. Learning (From) the                                                   
Electron Density: Transferability, Conformational and Chemical Diversity.                                                     
<i>Chimia</i> <b> 2020</b>, <i>74</i>, 232;                                                                                   
doi: 10.2533/chimia.2020.232                                                                                                  
</a>.                                                                                                                         
</li>                                       
</ol>
