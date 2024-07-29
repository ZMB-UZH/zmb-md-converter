# MD ImageXpress example data

Example data of the MD ImageXpress used at the ZMB.
The different .zip files correspond to the same data, exported in different ways.

## Data common to all zip-files:
- The folder *3420* contains data with:
  - 1 timepoint
  - 1 z-step
  - 1 well
  - 1 site
  - 1 channel
  
- The folder *3433* contains data with:
  - 1 timepoint
  - 3 z-steps
  - 2 wells
  - 2 sites
  - 2 channels

- The folder *3434* contains data with:
  - 1 timepoint
  - 3 z-steps
  - 2 wells
  - 2 sites
  - 4 channels:
    - w1: all z-planes
    - w2: only projection
    - w3: only 1 plane (0um offset)
    - w4: only 1 plane (10um offset)

- The folder *3435* contains data with:
  - 6 timepoints
  - 1 z-steps
  - 2 wells
  - 2 sites
  - 4 channels:
    - w1: all timepoints
    - w2: at first timepoint
    - w3: at first and last timepoint
    - w4: at every 3rd timepoint

- The folder *timeseries* contains three plates (*3433-3435*), corresponding to three timepoints with 1min intervals, each containing:
  - 1 timepoint
  - 3 z-steps
  - 2 wells
  - 2 sites
  - 2 channels

## Different export methods:

- *direct_transfer.zip*:  
  Data is directly transferred from the database, without using the MetaXpress software.
  Contains z-planes and z-projections.

- *INCarta_all-z.zip*:  
  Data is exported via the MetaXpress software to the INCarta file format.  
  Contains all z-planes.

- *INCarta_only-projection.zip*:  
  Data is exported via the MetaXpress software to the INCarta file format.  
  Contains only z-projections.

- *MetaXpress_all-z_include-projection.zip*:  
  Data is exported via the MetaXpress software to the MetaXpress file format.  
  Contains all z-planes and z-projections.

- *MetaXpress_all-z_no-projection.zip*:  
  Data is exported via the MetaXpress software to the MetaXpress file format.  
  Contains all z-planes.

- *MetaXpress_only-projection.zip*:  
  Data is exported via the MetaXpress software to the MetaXpress file format.  
  Contains only z-projections.

