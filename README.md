# IST Optimization and Algorithms Project

This repository contains the complete project for the Instituto Superior Técnico Optimization and Algorithms course.

## Authors
- David João Marafuz Gaspar - 106541
- Francisco Caetano Palma - 105949
- Pedro Gaspar Mónico - 106626
- Pedro Salazar Leite - 106812

## Project Overview

This project consists of multiple optimization tasks covering different problem types and solution approaches:

- **Optimal Control Problems**: Trajectory tracking and control optimization using convex optimization techniques
- **Circle Fitting Problems**: Geometric optimization using least squares and gradient descent methods
- **Multi-objective Optimization**: Trade-off analysis between competing objectives

The project explores various optimization techniques including convex optimization (CVXPY), least squares regression, and gradient descent algorithms.

## Repository Structure

```
ist-optimization-algorithms-project/
├── Development/                  # Development notebooks
│   ├── Data/                      # Dataset files
│   │   ├── target_*.npy         # Trajectory target data
│   │   ├── circle_data_*.npy    # Circle fitting data
│   │   └── dataset*.npy         # Additional datasets
│   ├── Task_1.ipynb             # Optimal control problems
│   ├── Task_4.ipynb             # Multi-trajectory optimization
│   ├── Task_7.ipynb             # Constrained optimization
│   ├── Task_9.ipynb             # Circle fitting via least squares
│   ├── Task_10.ipynb            # Circle fitting analysis
│   └── Task_11.ipynb            # Circle fitting with multi agent
│
├── Assignment.pdf               # Project specification
├── Report.pdf                   # Project report
└── README.md                    # This file
```

## Requirements

The project requires the following Python packages:

- `numpy` - Numerical computations and array operations
- `cvxpy` - Convex optimization library
- `matplotlib` - Visualization and plotting


## Usage

Each task is implemented as a Jupyter notebook in the `Development/` directory. To run a task:

1. Navigate to the `Development/` directory
2. Open the desired task notebook
3. Ensure the `Data/` directory is accessible from the notebook location
4. Run all cells to execute the optimization problem

All development work is documented in Jupyter notebooks within the `Development/` directory.

## Development Process

Each task follows a systematic approach:

1. **Problem Formulation** - Define variables, constraints, and objectives
2. **Model Setup** - Configure system dynamics and parameters
3. **Optimization** - Solve using appropriate method (CVXPY, least squares, gradient descent)
4. **Analysis** - Evaluate results and trade-offs
5. **Visualization** - Plot results and trajectories

## Documentation

- **Project Specification**: See `Assignment.pdf` for detailed problem descriptions
- **Project Report**: See `Report.pdf` for comprehensive analysis and results

## Important Notes

- All data files are stored in `Development/Data/`
- Notebooks assume data files are in a `Data/` subdirectory relative to the notebook location
- Results may vary slightly due to solver tolerances and numerical precision

---

*This is an IST Optimization and Algorithms course assignment for the academic year 2025–2026.*
