# Graphcast

This repository tries to mimic the inference step for encoder and processor of Google deepmind's Graphcast paper. Input features for nodes and edges for grids and meshes are randomly initialized for simplicity. Also the neighbour list for grid to mesh and mesh to mesh message passing are randomly initialized.

configure() function has not been used to input variable sizes. Instead the variables are provided in the .py file. The paper has the values for these variables (for example 103200). For running the code in colab without runtime issues, I divided those values using 10^4 (for example, modified value is 10).

e.g., real value: 103200, modified value : 10 #3200

# Run code :
```python 
python Graphcast_inference_py.py
```
or to understand the partwise operation, we also created different .py files in lib folder and you can check main.py to understand how those functions are called
```python 
python main.py
```
