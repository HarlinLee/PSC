# Principal Stiefel Coordinates (PSC)

Python code for the paper **[O(k)-Equivariant Dimensionality Reduction on Stiefel Manifolds](https://arxiv.org/abs/2309.10775)** by **Andrew Lee**, **Harlin Lee**, **Jose Perea**, **Nikolas Schonsheck**, and **Madeleine Weinstein**.

## What do I do first?

For the easiest example, 

```
git clone https://github.com/HarlinLee/PSC.git
cd PSC

conda create -n psc python=3.10.14
conda activate psc
pip install -r requirements/pip_essential.txt
python demo.py
```
If you are interested in reproducing the figures and tables in the paper, you would need to install more packages via `pip install -r requirements/pip.txt`.

## What's inside?
The `PSC/` package contains:
1. `projections.py`: Our PSC algorithm including manopt gradient descent and $\pi_\alpha$.  

2. `utils.py`: Helper code including calculation of projection error.

3. `data.py`: Generate data points on the Stiefel manifold.

4. `initilization.py`: Code for checking rank condition and optional RANSAC algorithm.

5. `comparison.py` and `plots.py`: Helper functions for PGA comparison.
 

The `experiments/` folder contains code to recreate experiments and figures produced in the paper. 

1. Section 5.1 `low_dim.ipynb`: Low-dimensional example with $k=1, n=2, N=3$ for PSC demonstration.

2. Section 5.2 `variance_comparison.ipynb`: Comparison with PGA. `var_data.pkl` is saved output. Check `comparison.py` and `plots.py` for more details.

3. Section 5.3 `neuron/`
    - `create_neural_response.ipynb`: A **Julia file** that generates data according to the neuronal stimulus space model. It uses auxilary functions in `julia_utilities.jl`. This makes files such as `neurons.h5`, `random_walk.h5` (or `100_neurons_13k_steps_nonuniform_half_random_walk.h5`), and `response_matrix.h5`. `centered_normalized_response_matrix_100_neurons_13k_steps_nonuniform_half_random_walk.h5` is the centered and $\ell_2$-normalized response matrix (see code at top of `stimulus_space_model.ipynb`).

    - `stimulus_space_model.ipynb`: Applies PSC, MDS, and persistent cohomology on the response matrix. Since MDS takes a while, `MDS_script.py` can be used as an alternative for running MDS inside the Jupyter notebook. Figures `psc_angle.pdf`, `psc_path.pdf`, `comparisons-others.pdf`, `comparisons-path.pdf` are generated, and dimensionality reduction results `circ_coords.h5`, `mds.h5`, `PSC.pkl` are saved.

  
4. Section 5.4 `brain.ipynb`: Brain connectivity matrix experiment. Uses matlab data saved in the folder `connectivity_matrices` and saves the plot `brain_projected.pdf`.


5. Section 5.5 `video/`
   - `video-clustering.ipynb`: Video clustering experiment.
  
6. Section 5.6 Data from nonlinear embeddings
   - Section 5.6.1. The M&ouml;bius bundle in `mobius_band.ipynb` and `run_trials_mobius_band.ipynb`.
   - Section 5.6.2. Whitney sums and the torus in `T2rank2embedding.ipynb`.
   - Section 5.6.3. The tangent bundle on $S^2$ in `sphere_tangent_bundle.ipynb` and `run_trials_sphere_tangent_bundle.ipynb`.
  
7. SM 1 and 3.1 `initialization.ipynb`. `runtime.npz` has saved runtime values. Check `initialization.py` for more details.

8. SM 3.3 `test_faster_PSC.ipynb`.
