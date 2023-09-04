

import numpy as np

import bilby

import lalsimulation as lalsim
import lal
import argparse
import os
import glob
import sys
from astropy.cosmology import z_at_value, Planck18
from astropy import units

def parse_cmd():
    parser=argparse.ArgumentParser()
    parser.add_argument("--psd_dir", help = "directory containing power spectral densities of detectors")
    args=parser.parse_args()
    return args
args=parse_cmd()

chirp_mass =  1.43
mass_ratio = 0.833 
a_1 = 0.
a_2 = 0.
tilt_1 = 0.
tilt_2 = 0.
phi_12 = 0.
luminosity_distance = 100
theta_jn = 0.1
phi_jl = 0.
psi=2.659
phase=1.3
ra = 5.445
dec = 0.0

z=z_at_value(Planck18.luminosity_distance,float(luminosity_distance)*units.Mpc).value
#orbs=arg.observing_run
approximant = 'TaylorF2ThreePointFivePN'
#eos=args.eos
psd_files=glob.glob(args.psd_dir+'*.txt')
#mc=(m1*m2)**(3./5.)/(m1+m2)**(1./5.)#>1.9 or (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
#mc* = (1.+z)
#print(mc)
#sys.exit()

outdir = '/home/jason/Bilby/my_own_runs/bilby_run02.1/outdir'+approximant+'/'+str(luminosity_distance)+'_'+str(int(chirp_mass*100)/100.)+'_'+str(int(chirp_mass*100)/100.)+'/'
if(not os.path.exists(outdir)):
    os.makedirs(outdir)

file_to_det={'H1':"aligo",'L1':'aligo','V1':'avirgo','K1':'kagra'}
duration =320
chirp_mass_min=0.92
chirp_mass_max=1.7

#outdir = 'pe_dir'
label = 'bns_example'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

minimum_frequency = 10
reference_frequency = 20
sampling_frequency = 4096.


#DICTIONARY FOR INJECTION VALUES
np.random.seed(88170235)
injection_parameters = dict(
    chirp_mass=chirp_mass,
    mass_ratio=mass_ratio,
    a_1=a_1, 
    a_2=a_2,
    tilt_1=tilt_1, 
    tilt_2=tilt_2, 
    theta_jn=theta_jn,
    luminosity_distance=luminosity_distance, 
    phi_jl=phi_jl,
    psi=psi, 
    phase=phase, 
    geocent_time=1264069376, 
    phi_12=phi_12,
    ra=ra, 
    dec=dec
)


#GENERATE A WAVEFORM
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=dict(
        waveform_approximant=approximant,
        reference_frequency=reference_frequency,
    )
)

#PSD TO INTERFEROMETER 
interferometers =bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1','K1'])

for ifo in interferometers:
    for fn in psd_files:
        if file_to_det[ifo.name] in fn:
            print(ifo.name,fn)
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=fn)

interferometers.set_strain_data_from_zero_noise(sampling_frequency, duration, start_time=injection_parameters['geocent_time'] - duration + 2.)
interferometers.inject_signal(
    parameters=injection_parameters,
    waveform_generator=waveform_generator
)

'''#PRIORS
priors = bilby.gw.prior.BNSPriorDict()
priors.pop("mass_1")
priors.pop("mass_2")

mc=injection_parameters['chirp_mass']
priors['chirp_mass'].minimum = mc * 0.95
priors['chirp_mass'].maximum = mc * 1.05
priors["mass_ratio"].minimum = 0.125
priors["mass_ratio"].maximum = 1
#priors['chi_1'] = bilby.core.prior.Uniform(minimum=-0.05, maximum=0.05, name="chi_1")
#priors['chi_2'] = bilby.core.prior.Uniform(minimum=-0.05, maximum=0.05, name="chi_2")
#priors['lambda_1'] = bilby.core.prior.Uniform(minimum=0.0, maximum=5000., name="lambda_1")
#priors['lambda_2'] = bilby.core.prior.Uniform(minimum=0.0, maximum=5000., name="lambda_1")
#priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,
    maximum=injection_parameters['geocent_time'] + 0.1,
    name='geocent_time', latex_label='$t_c$', unit='$s$'
)
priors["luminosity_distance"] = bilby.core.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=min(10,float(D)-10), maximum=max(100,float(D)+100), unit='Mpc', latex_label='$d_L$')
'''



#PRIORS
prior_dictionary = dict(
    chirp_mass=bilby.gw.prior.Uniform(name='chirp_mass', minimum=1.40, maximum=1.46),
    mass_ratio=bilby.gw.prior.Uniform(name='mass_ratio', minimum=0.25, maximum=1),
    mass_1=bilby.gw.prior.Constraint(name='mass_1', minimum=1.0, maximum=3.0),
    mass_2=bilby.gw.prior.Constraint(name='mass_2', minimum=1.0, maximum=3.0),
    a_1=bilby.gw.prior.Uniform(name='a_1', minimum=0, maximum=0.05,
                               latex_label='$a_1$', unit=None, boundary=None),
    a_2=bilby.gw.prior.Uniform(name='a_2', minimum=0, maximum=0.05,
                               latex_label='$a_2$', unit=None, boundary=None),
    tilt_1=bilby.core.prior.DeltaFunction(peak=0.0),
    tilt_2=bilby.core.prior.DeltaFunction(peak=0.0),
    phi_12=bilby.core.prior.DeltaFunction(peak=0.0),
    phi_jl=bilby.gw.prior.Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi,
                                  boundary='periodic', latex_label='$\\phi_{JL}$', unit=None),
    luminosity_distance=bilby.core.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=min(10,float(luminosity_distance)-10), maximum=max(120,float(luminosity_distance)+120), unit='Mpc', latex_label='$d_L$'),
    dec=bilby.core.prior.DeltaFunction(peak=0.0),
    ra=bilby.core.prior.DeltaFunction(peak=5.445),
    theta_jn=bilby.prior.Sine(name='theta_jn', latex_label='$\\theta_{JN}$',
                              unit=None, minimum=0, maximum=np.pi, boundary=None),
    psi=bilby.gw.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic',
                               latex_label='$\\psi$', unit=None)
)

priors = bilby.gw.prior.BBHPriorDict(dictionary=prior_dictionary)

# set a small margin on time of arrival
priors['geocent_time'] = bilby.core.prior.DeltaFunction(
    peak=1264069376
)


#ROQ LIKELIHOOD
search_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.binary_neutron_star_roq,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=dict(
        waveform_approximant=approximant,
        reference_frequency=reference_frequency
    )
)
roq_params = np.array(
    [(minimum_frequency, sampling_frequency / 2, duration, chirp_mass_min, chirp_mass_max, 0)],
    dtype=[("flow", float), ("fhigh", float), ("seglen", float), ("chirpmassmin", float), ("chirpmassmax", float), ("compmin", float)]
)
likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient(
    interferometers, 
    search_waveform_generator,
    priors,
    linear_matrix="/home/jason/Bilby/my_own_runs/bilby_run02.1/roq/basis_256s.hdf5", 
    quadratic_matrix="/home/jason/Bilby/my_own_runs/bilby_run02.1/roq/basis_256s.hdf5",
    roq_params=roq_params,
    distance_marginalization=True, 
    phase_marginalization=True
)

# SAMPLING
npool = 100
nact = 10
nlive = 2000
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler='dynesty',
    use_ratio=True,
    nlive=nlive,
    walks=100,
    maxmcmc=5000,
    nact=nact,
    npool=npool,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
    result_class=bilby.gw.result.CBCResult,
)   


result.plot_corner()
