# DeepEF: A Thermodynamic Hypothesis-Based GNN for Protein Free Energy Estimation

### Overview
**DeepEF** is a Graph Neural Network (GNN) that approximates protein free energy by leveraging the thermodynamic hypothesis of protein folding as its foundation. According to the thermodynamic hypothesis, a protein's native structure corresponds to the global minimum of its free energy landscape. Proposed by Anfinsen in the 1960s, this hypothesis has become a cornerstone in structural biology, providing a framework for interpreting protein stability and folding dynamics.

### Problem Statement
The prediction of protein stability and free energy is a fundamental challenge in biochemistry and structural biology. Existing methods often rely on experimental stability data and transfer learning approaches, which may limit generalizability. DeepEF addresses this by constructing a physics-inspired, data-driven model that relies on fundamental thermodynamic principles, offering the potential for broader applicability in stability prediction across diverse protein structures.

### Approach
DeepEF operates using synthetic data generated based on corollaries of the thermodynamic hypothesis. Without direct experimental energy measurements, this approach uses physics-based approximations to guide the model. Key components include:

- **Input**: Protein sequences and structural representations.
- **Output**: An energy value associated with each protein, approximating its free energy.
- **Synthetic Data**: Includes extended chains and permuted sequences to simulate various structural conditions.
- **Loss Function**: Incorporates thermodynamic corollaries that enforce expected energy differences, gradient behavior, and stability dynamics.

### Key Features
- **Thermodynamic Hypothesis Foundation**: DeepEF is designed to respect thermodynamic principles, predicting lower energy for native structures versus modified sequences or structures.
- **Synthetic Loss Function**: Based on the hypothesis, the loss function enforces properties such as:
  - Wild-type sequences in native conformations have minimal energy.
  - Non-native conformations exhibit higher energy.
  - Energy gradient approaches zero at the native conformation.
- **Extrapolative Capability**: DeepEF generalizes to predict stability measures that extend beyond the synthetic dataset, closely mirroring experimental stability observations.

### Significance
To our knowledge, DeepEF is the first model to incorporate thermodynamic principles as a foundation for protein stability prediction, marking a significant step towards physics-inspired deep learning models in structural biology. Its design enables it to predict stability outcomes beyond the training domain, providing an innovative tool for protein engineering, variant stability assessment, and structural biology research.

