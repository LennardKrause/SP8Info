# SP8Info

SPring-8 Experiment Info
A tool to optimize the primary beam attenuation factor to
stay within the linear response region of the detector

### Usage:
 - Perform a 180 degree scan
   Scan width: 0.1 degrees
   Exposure time: 0.1 seconds
 - Run the program PEInfo.py again
 - Give the image path as argument (-f path/to/images)

## PROGRAM OUTPUT

### Experiment Info (weak scatterers):
 - Finds maximum intensity of given images
 - Crudely estimates the maximum flux
 - Simply extrapolates an exposure time
 - Suggests a suitable attenuation factor
 - Estimated attenuation sufficient for *weak* scatterers
 - Writes a phi.log file that you can use for a fine-sliced
   rocking scan over the strongest peak to allow a better
   estimation of both the flux and the attenuation factor
 - A peak profile should not exceed 3.0 degrees

### Fine-Slice Info (strong scatterers):
 - Fits the fine-sliced reflection profile assuming a
   Gaussian model to estimate the peak flux
 - Better estimate of maximum flux
 - Goodness of Fit should be 95 % or higher!
   Otherwise try to increase profiling queue (-q 25)
 - Recommended for *strongly* scattering samples
 
### The program will suggest parameters:
 - Attenuator:    Name of the attenuator to use
 - Transmission:  Resulting beam intensity [%]
 - Utilization:   Est. flux relative to linear region [%]
 - Exposure time: Estimated exposure time [seconds/image]

## GENERAL INFORMATION

### PILATUS3 X 1M CdTe:
 - Image bit depth:    32 bit
 - Readout bit depth:  20 bit
 - Overflow state:     1,048,575
 - Maximum count rate: 1*10^7 photons/s/pixel
 - Linear region:      < 600,000 photons/s/pixel
 - Readout time:       0.95 ms (passive)
 - The duty cycle (passive/active) should remain above 99 %
   so the shortest reasonable exposure time is 0.1 seconds.
   This holds for any data collection but not for the
   fine-slice experiment where we are only interested in
   modelling the peak flux
 - Exposure time [s] > Duty cycle
   - 1.00 -> 99.91 %
   - 0.50 -> 99.81 %
   - 0.10 -> 99.05 %
   - 0.05 -> 98.10 %
   - 0.01 -> 90.50 %
