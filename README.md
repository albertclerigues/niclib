# niclib v0.3

Niclib is a library for neuroimaging and deep learning, it includes specific utility functions and components to ease the implementation and design of deep learning pipelines. 

#### niclib
- GPU selection management
- List utils (splitting, resampling, indexing...)
- Time utils (timestamping, time formatting...)
- File utils (load/save .csv, .txt, ...)

#### data
- Data utils (padding, contrast and range operations, border cropping, histogram matching...)
- Nifti utils (load/save, reorient, ...)
- Generators
  - Patch/Slice generators

#### net
- Predefined models (UnetRonneberger, UnetGuerrero, UNetCicek...) 
- Loss functions (Focal loss, dice loss, GDL, …) 
- Training
  - Built-in metrics and plugin system
- Checkpoints (load, save, ...) 

#### eval
- Metrics (DSC, MAE, MSE, SSIM ...)

