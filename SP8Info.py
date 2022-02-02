import os, sys, re, glob, argparse, logging
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

def init_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', required=True,  default='',   metavar='dir',  type=str,   dest='_PATH',    help='Path to image directory')
    parser.add_argument('-n', required=False, default=1800, metavar='1800', type=int,   dest='_NIMG',    help='Number of image to load')
    parser.add_argument('-q', required=False, default=15,   metavar='15',   type=int,   dest='_QUEUE',   help='Use 2N+1 Images for peak profiling, skips the first/last images')
    parser.add_argument('-s', required=False, default=3.0,  metavar='3.0',  type=float, dest='_SWEEP',   help='Scan sweep range for the fine slicing experiment')
    parser.add_argument('-i', required=False, default=0.01, metavar='0.01', type=float, dest='_INCRE',   help='Image width increment for the fine slicing')
    parser.add_argument('-m', required=False, default=6e5,  metavar='6e5',  type=int,   dest='_MCPS',    help='Maximum linear counts/second')
    parser.add_argument('-o', required=False, default=8e5,  metavar='8e5',  type=int,   dest='_OCPP',    help='Optimal intensity [counts/pixel]')
    parser.add_argument('-l', required=False, action='store_true',                      dest='_LORENTZ', help='Fit Lorentzian instead of Gaussian to peak [Flag]?')
    parser.add_argument('-w', required=False, action='store_true',                      dest='_WEIGHTS', help='Use sqrt(I) as weights for fit [Flag]?')
    # print help if script is run without arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        raise SystemExit
    return parser.parse_args()

def show_intro():
    print('''
 SPring-8 Experiment Info
 A tool to optimize the primary beam attenuation factor to
 stay within the linear response region of the detector
 
 Usage:
  - Perform a 180 degree scan
    Scan width: 0.1 degrees
    Exposure time: 0.1 seconds
  - Run the program PEInfo.py again
  - Give the image path as argument (-f path/to/images)
 
 PROGRAM OUTPUT
 
 Experiment Info (weak scatterers):
  - Finds maximum intensity of given images
  - Crudely estimates the maximum flux
  - Simply extrapolates an exposure time
  - Suggests a suitable attenuation factor
  - Estimated attenuation sufficient for *weak* scatterers
  - Writes a phi.log file that you can use for a fine-sliced
    rocking scan over the strongest peak to allow a better
    estimation of both the flux and the attenuation factor
  - A peak profile should not exceed 3.0 degrees
 
 Fine-Slice Info (strong scatterers):
  - Fits the fine-sliced reflection profile assuming a
    Gaussian model to estimate the peak flux
  - Better estimate of maximum flux
  - Goodness of Fit should be 95 % or higher!
    Otherwise try to increase profiling queue (-q 25)
  - Recommended for *strongly* scattering samples
  
 The program will suggest parameters:
  - Attenuator:    Name of the attenuator to use
  - Transmission:  Resulting beam intensity [%]
  - Utilization:   Est. flux relative to linear region [%]
  - Exposure time: Estimated exposure time [seconds/image]
 
 GENERAL INFORMATION
 
 PILATUS3 X 1M CdTe:
  - Image bit depth:    32 bit
  - Readout bit depth:  20 bit
  - Overflow state:     1,048,575
  - Maximum count rate: 1*10^7 photons/s/pixel
  - Linear region:      < 600,000 photons/s/pixel
  - Readout time:       0.95 ms (passive)
  - The duty cycle (passive/active) should remain above 99 %
    so the shortest reasonable exposure time is 0.1 seconds
  - This holds for any data collection but not for the
    fine-slice experiment where we are only interested in
    modelling the peak flux
  - Exposure time > Duty cycle
    1.00 seconds  > 99.91 %
    0.50 seconds  > 99.81 %
    0.10 seconds  > 99.05 %
    0.05 seconds  > 98.10 %
    0.01 seconds  > 90.50 %
 ''')

def read_Pilatus3X1M(fname):
    with open(fname, 'rb') as b:
        # we are not interested in the header
        #head = b.read(4096).decode('unicode_escape')
        b.seek(4096)
        data = np.ndarray(shape=(1043, 981), dtype=np.int32, buffer=b.read(4092732))
    return data

def max_on_image(fpath):
    data = read_Pilatus3X1M(fpath)
    return data.max()

def analyse_pre_experiment(_ARGS, att_dict, img_pth, img_exp, img_wth, max_i, max_x, max_y, max_ang):
    # Maximum counts per second
    max_cps = int(max_i/img_exp)
    # Current flux load
    cps_load = _ARGS._MCPS / max_cps
    # Find best attenuator by value
    att_val = min(att_dict.keys(), key=lambda x:abs(x-cps_load))
    # Figure out a better exposure
    sug_exp = (_ARGS._OCPP/max_cps)/att_val
    # Log it
    logging.info('\n> Experiment Info')
    logging.info('  Maximum Image: {}'.format(os.path.splitext(os.path.basename(img_pth))[0]))
    logging.info('  Exposure time:{:12.2f} seconds'.format(img_exp))
    logging.info('  Scan width:   {:12.2f} degrees'.format(img_wth))
    logging.info('  Maximum I:    {:12,} counts/pixel'.format(max_i))
    logging.info('  Maximum CPS:  {:12,} counts/second'.format(max_cps))
    #logging.info('  Maximum XY:   {:>12} coordinates'.format('{:>4} {}'.format(max_x, max_y)))
    logging.info('  Maximum Flux: {:>12.0f} % of linear maximum'.format(round(1/cps_load*100,0)))
    #logging.info('\n> Suggested Exposure at Transmission [~50 keV]')
    #logging.info('  {:5} [{:4} / {:^4}]: {:>14}'.format('Att', 'Flux', 'Max', 'Exposure time'))
    #for entry in sorted(att_dict, reverse=True):
    #    sug_exp = (_ARGS._OCPP/max_cps)/entry
    #    if entry/cps_load < 1.25:
    #        logging.info('  {:5} [{:3.0f}% / {:3.0f}%]: {:8.2f} s/img'.format(att_dict[entry], round(entry*100,0), round(entry/cps_load*100,0), sug_exp))
    
    # Get the name
    att_nam = att_dict[att_val]
    # Estimate flux relative to linear region
    cps_rel = att_val/cps_load
    # Show suggested parameters
    logging.info('\n> Suggested Parameters [~50 keV]')
    logging.info('  Attenuator:    {:>6}'.format(att_nam))
    logging.info('  Transmission:  {:6.0f} %'.format(att_val*100))
    logging.info('  Utilization:   {:6.0f} %'.format(cps_rel*100))
    logging.info('  Exposure time: {:6.2f} seconds/image'.format(sug_exp))
    
    # Sweep in degrees for fine-slice
    fin_swe = _ARGS._SWEEP
    # Starting angle for fine-slice
    fin_sta = max_ang - fin_swe/2
    # Fine-slice angle increment per image
    fin_inc = _ARGS._INCRE
    # Total number of images
    fin_num = int(fin_swe/fin_inc)
    # Fine-slice exposure time
    fin_exp = img_exp/img_wth*fin_inc
    # Fine-Slice Experiment, show info and write the file
    logging.info('\n> Fine-Slice Parameters')
    logging.info('  Angle:     {:8.2f} degrees'.format(fin_sta))
    logging.info('  Sweep:     {:8.2f} degrees'.format(fin_swe))
    logging.info('  Increment: {:8.2f} degrees'.format(fin_inc))
    logging.info('  Exposure:  {:8.2f} seconds'.format(fin_exp))
    logging.info('  Images:    {:8}'.format(fin_num))
    # Write it
    with open('{}_phi.log'.format(os.path.split(img_pth)[0]),'w') as wf:
        wf.write('Equal\n')
        wf.write('sec\n')
        wf.write('1\n')
        wf.write('Omega\n')
        wf.write('0 N2\n')
        wf.write('1\n')
        wf.write('1\n')
        wf.write('{:8}\n'.format(fin_num))
        wf.write('tif\n')
        wf.write('1\n')
        wf.write('1\n')
        wf.write('1\n')
        wf.write('0\n')
        wf.write('0.50\n')
        wf.write('{:8.3f} {:8.3f} {:8.3f}    0.000    0.000    0.000    0.000    0.000    0.000    0.000\n'.format(fin_sta, fin_swe, fin_exp))
    logging.info('\n> {}_phi.log written!'.format(os.path.basename(os.path.split(img_pth)[0])))
    
def analyse_fine_slice(_ARGS, att_dict, path_list, img_pth, img_idx, img_exp, img_wth, max_i, max_x, max_y):
    # Import needed only for the fine slice analysis
    from scipy.optimize import curve_fit
    from scipy.special import erf
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    import matplotlib.pyplot as plt
    
    def I_gauss(x,I,mu,sig):
        return I*(1/2.*(1+erf(((x+dx/2.)-mu)/(sig*np.sqrt(2)))) - 1/2.*(1+erf(((x-dx/2.)-mu)/(sig*np.sqrt(2)))))
    
    def f_gauss(t,I,mu,sig):
        return I*(1/((sig/v)*np.sqrt(2*np.pi))*np.exp(-1/2.*(t-mu/v)**2/(sig/v)**2))
    
    def f_gauss_ang(t,I,mu,sig):
        return I*(1/((sig)*np.sqrt(2*np.pi))*np.exp(-1/2.*(t-mu)**2/(sig)**2))
    
    def I_lorentz(x,I,mu,gamma):
        return I*(1/np.pi*np.arctan2((x+dx/2.)-mu,gamma) + 1/2. - (1/np.pi*np.arctan2((x-dx/2.)-mu,gamma) + 1/2.))
        
    def f_lorentz(t,I,mu,gamma):
        return I*1/np.pi*(gamma/v)/((t-mu/v)**2+(gamma/v)**2)
    
    # Full queue size
    fit_fqs = 2 * _ARGS._QUEUE + 1
    # Grid stepsize
    fit_stp = _ARGS._QUEUE * float(img_wth)
    # Stepping grid in frames
    fit_grf = np.linspace(-fit_stp, fit_stp, fit_fqs)
    # Rotation speed: deg/sec
    img_ros = img_wth / img_exp
    # Stepping grid in seconds
    fit_grs = np.linspace(fit_grf[0]/img_ros, fit_grf[-1]/img_ros, len(fit_grf)*10)
    # re-assignment to be used by the fitting functions
    dx = img_wth
    v  = img_ros
    # Log exp parameters
    logging.info('\n> Experiment Info')
    logging.info('  Exposure time: {:12.2f} seconds'.format(img_exp))
    logging.info('  Scan width:    {:12.2f} degrees'.format(img_wth))
    logging.info('  Maximum on {} '.format(os.path.basename(img_pth)))
    # Get the spot profile
    logging.info('\n> Spot Profile ({})'.format(_ARGS._QUEUE))
    logging.info('  Profile [counts/pixel]')
    img_start = img_idx - _ARGS._QUEUE
    if img_start < 0:
        logging.info('Error profile out of bounds!')
        raise SystemExit
    img_end   = img_idx + _ARGS._QUEUE + 1
    pro_lst = []
    for idx, fpath in enumerate(path_list[img_start:img_end]):
        data = read_Pilatus3X1M(fpath)
        pro_int = data[max_y, max_x]
        pro_lst.append(pro_int)
        logging.info('  Profile > {:12,}'.format(pro_int))
    pro_dat = np.asarray(pro_lst)
    
    max_cps = int(max_i/img_exp)
    pro_sum = np.sum(pro_dat)
    p0 = [2*pro_sum, 0, 0.2]
    
    use_weights = False
    
    if _ARGS._WEIGHTS:
        weights = np.sqrt(pro_dat)
    else:
        weights = np.ones(len(pro_dat))
    
    # sigma or gamma (gauss / lorentz)
    try:
        if _ARGS._LORENTZ:
            g_popt, g_pcov = curve_fit(I_lorentz, fit_grf, pro_dat, p0=p0, sigma=weights)
        else:
            g_popt, g_pcov = curve_fit(I_gauss, fit_grf, pro_dat, p0=p0, sigma=weights)
            # residual sum of squares
            ss_res = np.sum((pro_dat - I_gauss(fit_grf, *g_popt)) ** 2)
            # total sum of squares
            ss_tot = np.sum((pro_dat - np.mean(pro_dat)) ** 2)
            # r-squared
            r2 = 1 - (ss_res / ss_tot)
    except RuntimeError:
        logging.info('RuntimeError: Optimal parameters not found!')
        raise
    
    flux_est = g_popt[0]/((g_popt[2]/img_ros)*np.sqrt(2*np.pi))
    if _ARGS._LORENTZ:
        logging.info('\n> Fitted Peak (Lorentzian, weighted: {})'.format(_ARGS._WEIGHTS))
    else:
        logging.info('\n> Fitted Peak (Gaussian, weighted: {})'.format(_ARGS._WEIGHTS))
        logging.info('  [fit] Goodness of Fit R2:   {:14,.2f} %'.format(r2*100))
        logging.info('  [fit] Estimated FWHM:       {:14,.2f} degree'.format(g_popt[2]*2.355))
    logging.info('  [fit] Estimated Flux:       {:14,.2f} Photons/second'.format(flux_est))
    logging.info('  [fit] Integrated Intensity: {:14,.2f} Photons'.format(g_popt[0]))
    logging.info('  [pix] Normalized Counts:    {:14,.2f} Photons/second'.format(max_cps))
    logging.info('  [pix] Integrated Intensity: {:14,.2f} Photons'.format(pro_sum))
    
    # Current flux load
    cps_load = _ARGS._MCPS / max(flux_est, max_cps)
    # Find best attenuator by value
    att_val = min(att_dict.keys(), key=lambda x:abs(x-cps_load))
    # Get the name
    att_nam = att_dict[att_val]
    # Figure out a better exposure
    sug_exp = (_ARGS._OCPP/max_cps)/att_val
    # Estimate flux relative to linear region
    cps_rel = att_val/cps_load
    # Show suggested parameters
    logging.info('\n> Suggested Parameters [~50 keV]')
    logging.info('  Attenuator:    {:>6}'.format(att_nam))
    logging.info('  Transmission:  {:6.0f} %'.format(att_val*100))
    logging.info('  Utilization:   {:6.0f} %'.format(cps_rel*100))
    logging.info('  Exposure time: {:6.2f} seconds/image'.format(sug_exp))
    # Set plot parameters
    SMALL_SIZE = 8
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 8
    plt.rc('font', family='sans-serif')
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # init plot
    fig, (p11,p12) = plt.subplots(1,2)
    fig.suptitle('{}'.format(os.path.basename(img_pth)), weight='bold')
    fig.set_size_inches(3.484252, 2.1776575)
    plt.subplots_adjust(left   = 0.15, # the left side of the subplots of the figure
                        right  = 0.98, # the right side of the subplots of the figure
                        bottom = 0.18, # the bottom of the subplots of the figure
                        top    = 0.85, # the top of the subplots of the figure
                        wspace = 0.50, # the amount of width reserved for space between subplots,
                                       # expressed as a fraction of the average axis width
                        hspace = 0.50, # the amount of height reserved for space between subplots,
                                       # expressed as a fraction of the average axis height
                        )
    p11.set_xlabel('Scan width [°]')
    p11.set_ylabel('Counts per pixel [Photons]')
    p12.set_xlabel('Time [s]')
    p12.set_ylabel('Flux [Photons/s]')
    p11.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    p12.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if _ARGS._LORENTZ:
        gf = p11.plot(fit_grf, I_lorentz(fit_grf, *g_popt), '-', lw=2.0, color='#37a0cb', label='Fit'.format(g_popt[0]))
        gs = p12.plot(fit_grs, f_lorentz(fit_grs, *g_popt), '-', lw=2.0, color='#37a0cb', label=''.format(flux_est))
    else:
        gf = p11.plot(fit_grf, I_gauss(fit_grf, *g_popt), '-', lw=2.0, color='#37a0cb', label='Fit'.format(g_popt[0]))
        gs = p12.plot(fit_grs, f_gauss(fit_grs, *g_popt), '-', lw=2.0, color='#37a0cb', label=''.format(flux_est))
    data = p11.plot(fit_grf, pro_dat, 'k*', ms=4, label='Data')
    plt_nam = '{}.pdf'.format(os.path.split(img_pth)[0])
    plt.savefig(plt_nam)
    plt.close()
    logging.info('\n> {} written!'.format(os.path.basename(plt_nam)))

def main():
    # Transmission : Attenuator (50.0 keV)
    # This table needs to be adjusted/expanded
    # upon change of the X-ray energy
    att_dict = {0.05:'Ta350',
                0.12:'Ta250',
                0.31:'Ni600',
                0.45:'Ni400',
                0.65:'Cu200',
                1.00:'None'}
    
    # Print intro text
    show_intro()
    
    # Parse arguments
    _ARGS = init_argparser()
    
    # Find images
    path_list = sorted(glob.glob(os.path.join(_ARGS._PATH,'*.tif')))[:_ARGS._NIMG]
    if len(path_list) == 0:
        print('No *.tif files found!')
        raise SystemExit
    image_list = path_list[_ARGS._QUEUE:-_ARGS._QUEUE]
    
    # Init logging
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[logging.FileHandler('{}.lst'.format(_ARGS._PATH),'w'),
                                  logging.StreamHandler()])
    logging.info('> Analyzing Images:'.format(len(path_list)))
    logging.info('  {} .tif files found'.format(len(path_list)))
    
    # Find the maximum
    with Pool() as p:
        result = p.map(max_on_image, tqdm(image_list))
    # Get max of images
    img_max = max(result)
    # Index of max image (will only return first max!)
    img_idx = result.index(img_max)
    # Path to max image
    img_pth = image_list[img_idx]
    # Read the max image again
    img_dat = read_Pilatus3X1M(img_pth)
    if img_max != img_dat.max():
        logging.info('Something went wrong!')
        raise SystemExit
    # X and Y of max
    max_x, max_y = np.array(np.unravel_index(np.nanargmax(img_dat, axis=None), img_dat.shape))[::-1]
    
    # Get the scan range from .inf
    with open(''.join([os.path.splitext(img_pth)[0], '.inf'])) as inf:
        img_sta, img_end, img_wth, img_exp = list(map(float, re.findall('\s*SCAN_ROTATION\s*=\s*(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', inf.read())[0]))
    max_ang = (img_sta + img_end) / 2
    
    if img_wth <= 0.01:
        logging.info('\n> Analysing: Fine-Slice Experiment')
        analyse_fine_slice(_ARGS, att_dict, image_list, img_pth, img_idx, img_exp, img_wth, img_max, max_x, max_y)
    else:
        analyse_pre_experiment(_ARGS, att_dict, img_pth, img_exp, img_wth, img_max, max_x, max_y, max_ang)
    
if __name__ == '__main__':
    main()